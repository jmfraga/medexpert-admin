"""
MedExpert Admin - Database Module
SQLite database for experts, clients, web sources, licenses, and usage tracking.
"""

import sqlite3
import json
import secrets
from datetime import datetime
from pathlib import Path
from rich.console import Console

console = Console()

DB_PATH = Path("data/medexpert_admin.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS experts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            slug TEXT UNIQUE NOT NULL,
            icon TEXT DEFAULT '&#9678;',
            system_prompt TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            hostname TEXT UNIQUE NOT NULL,
            tailscale_ip TEXT DEFAULT '',
            plan TEXT DEFAULT 'basico' CHECK(plan IN ('basico', 'profesional', 'enterprise')),
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'inactive', 'suspended')),
            license_key TEXT UNIQUE,
            license_expires TEXT,
            max_sessions_per_day INTEGER DEFAULT 50,
            notes TEXT DEFAULT '',
            client_config TEXT DEFAULT '{}',
            last_seen TEXT DEFAULT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS client_experts (
            client_id INTEGER NOT NULL,
            expert_id INTEGER NOT NULL,
            last_synced TEXT DEFAULT NULL,
            chromadb_version TEXT DEFAULT '',
            config_version TEXT DEFAULT '',
            PRIMARY KEY (client_id, expert_id),
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
            FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS web_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expert_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            url TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'public'
                CHECK(source_type IN ('public', 'monitor', 'authenticated')),
            category TEXT DEFAULT '',
            status TEXT DEFAULT 'active'
                CHECK(status IN ('active', 'error', 'update_available', 'disabled')),
            current_version TEXT DEFAULT '',
            last_fetched TEXT DEFAULT NULL,
            last_checked TEXT DEFAULT NULL,
            content_hash TEXT DEFAULT NULL,
            session_cookie TEXT DEFAULT NULL,
            css_selector_content TEXT DEFAULT '',
            css_selector_version TEXT DEFAULT '',
            version_regex TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            error_message TEXT DEFAULT '',
            crawl_depth INTEGER DEFAULT 0,
            url_pattern TEXT DEFAULT '',
            login_url TEXT DEFAULT '',
            login_username TEXT DEFAULT '',
            login_password TEXT DEFAULT '',
            url_exclude TEXT DEFAULT '',
            use_browser INTEGER DEFAULT 0,
            allowed_domains TEXT DEFAULT '',
            min_content_length INTEGER DEFAULT 2000,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE,
            UNIQUE(expert_id, url)
        );

        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL DEFAULT '',
            provider TEXT NOT NULL DEFAULT 'anthropic',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL DEFAULT '',
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS distribution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            expert_slug TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('chromadb_push', 'config_push', 'license_push')),
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'success', 'error')),
            details TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE
        );
    """)
    conn.commit()

    # Migrations
    for col, defn in [
        ("use_browser", "INTEGER DEFAULT 0"),
        ("allowed_domains", "TEXT DEFAULT ''"),
        ("min_content_length", "INTEGER DEFAULT 2000"),
    ]:
        try:
            conn.execute(f"ALTER TABLE web_sources ADD COLUMN {col} {defn}")
            conn.commit()
        except Exception:
            pass  # Column already exists
    # Migrations
    try:
        conn.execute("ALTER TABLE clients ADD COLUMN client_config TEXT DEFAULT '{}'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    conn.close()
    console.print("[green]Database initialized[/green]")


# ─────────────────────────────────────────────
# Experts CRUD
# ─────────────────────────────────────────────

def create_expert(name: str, slug: str, system_prompt: str, icon: str = "&#9678;") -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO experts (name, slug, system_prompt, icon) VALUES (?, ?, ?, ?)",
            (name, slug, system_prompt, icon),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_experts() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM experts ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_expert_by_id(expert_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM experts WHERE id = ?", (expert_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_expert_by_slug(slug: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM experts WHERE slug = ?", (slug,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_expert(expert_id: int, name: str = None, system_prompt: str = None,
                  icon: str = None):
    conn = get_connection()
    updates, params = [], []
    if name is not None:
        updates.append("name = ?"); params.append(name)
    if system_prompt is not None:
        updates.append("system_prompt = ?"); params.append(system_prompt)
    if icon is not None:
        updates.append("icon = ?"); params.append(icon)
    if updates:
        params.append(expert_id)
        conn.execute(f"UPDATE experts SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()


def delete_expert(expert_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM experts WHERE id = ?", (expert_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ─────────────────────────────────────────────
# Clients CRUD
# ─────────────────────────────────────────────

def create_client(name: str, hostname: str, plan: str = "basico",
                  tailscale_ip: str = "", notes: str = "") -> int:
    license_key = "mel_" + secrets.token_hex(16)
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO clients (name, hostname, plan, tailscale_ip, license_key, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name, hostname, plan, tailscale_ip, license_key, notes))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_clients() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM clients ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_client_by_id(client_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM clients WHERE id = ?", (client_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_client_by_hostname(hostname: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM clients WHERE hostname = ?", (hostname,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_client(client_id: int, **kwargs):
    conn = get_connection()
    allowed = {"name", "hostname", "plan", "tailscale_ip", "status",
               "license_expires", "max_sessions_per_day", "notes", "last_seen",
               "client_config"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            updates.append(f"{key} = ?"); params.append(val)
    if updates:
        params.append(client_id)
        conn.execute(f"UPDATE clients SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()


def delete_client(client_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM clients WHERE id = ?", (client_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ─────────────────────────────────────────────
# Client-Expert assignments
# ─────────────────────────────────────────────

def assign_expert_to_client(client_id: int, expert_id: int):
    conn = get_connection()
    conn.execute("""
        INSERT OR IGNORE INTO client_experts (client_id, expert_id)
        VALUES (?, ?)
    """, (client_id, expert_id))
    conn.commit()
    conn.close()


def remove_expert_from_client(client_id: int, expert_id: int):
    conn = get_connection()
    conn.execute(
        "DELETE FROM client_experts WHERE client_id = ? AND expert_id = ?",
        (client_id, expert_id),
    )
    conn.commit()
    conn.close()


def get_client_experts(client_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT e.*, ce.last_synced, ce.chromadb_version, ce.config_version
        FROM experts e
        JOIN client_experts ce ON e.id = ce.expert_id
        WHERE ce.client_id = ?
        ORDER BY e.name
    """, (client_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_client_expert_sync(client_id: int, expert_id: int,
                               chromadb_version: str = None, config_version: str = None):
    conn = get_connection()
    updates = ["last_synced = datetime('now')"]
    params = []
    if chromadb_version:
        updates.append("chromadb_version = ?"); params.append(chromadb_version)
    if config_version:
        updates.append("config_version = ?"); params.append(config_version)
    params.extend([client_id, expert_id])
    conn.execute(
        f"UPDATE client_experts SET {', '.join(updates)} WHERE client_id = ? AND expert_id = ?",
        params,
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Web Sources CRUD (same as monolith)
# ─────────────────────────────────────────────

def create_web_source(expert_id: int, name: str, url: str, source_type: str = "public",
                      category: str = "", css_selector_content: str = "",
                      css_selector_version: str = "", version_regex: str = "",
                      notes: str = "", crawl_depth: int = 0, url_pattern: str = "",
                      login_url: str = "", login_username: str = "",
                      login_password: str = "", url_exclude: str = "",
                      use_browser: int = 0,
                      allowed_domains: str = "",
                      min_content_length: int = 2000) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO web_sources (expert_id, name, url, source_type, category,
                                     css_selector_content, css_selector_version,
                                     version_regex, notes, crawl_depth, url_pattern,
                                     login_url, login_username, login_password,
                                     url_exclude, use_browser, allowed_domains,
                                     min_content_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (expert_id, name, url, source_type, category,
              css_selector_content, css_selector_version, version_regex, notes,
              crawl_depth, url_pattern, login_url, login_username, login_password,
              url_exclude, use_browser, allowed_domains, min_content_length))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_web_sources_for_expert(expert_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM web_sources WHERE expert_id = ? ORDER BY created_at",
        (expert_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_web_source_by_id(source_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM web_sources WHERE id = ?", (source_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_web_source_status(source_id: int, status: str, error_message: str = "",
                             current_version: str = None, content_hash: str = None,
                             last_fetched: str = None, last_checked: str = None):
    conn = get_connection()
    try:
        updates = ["status = ?", "error_message = ?"]
        params = [status, error_message]
        if current_version is not None:
            updates.append("current_version = ?"); params.append(current_version)
        if content_hash is not None:
            updates.append("content_hash = ?"); params.append(content_hash)
        if last_fetched is not None:
            updates.append("last_fetched = ?"); params.append(last_fetched)
        if last_checked is not None:
            updates.append("last_checked = ?"); params.append(last_checked)
        params.append(source_id)
        conn.execute(f"UPDATE web_sources SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def update_web_source_cookie(source_id: int, session_cookie: str):
    conn = get_connection()
    try:
        conn.execute("UPDATE web_sources SET session_cookie = ? WHERE id = ?",
                     (session_cookie, source_id))
        conn.commit()
    finally:
        conn.close()


def update_web_source_credentials(source_id: int, login_url: str = "",
                                   login_username: str = "", login_password: str = "",
                                   crawl_depth: int = None, url_pattern: str = None,
                                   url_exclude: str = None):
    conn = get_connection()
    try:
        updates = ["login_url = ?", "login_username = ?", "login_password = ?"]
        params = [login_url, login_username, login_password]
        if crawl_depth is not None:
            updates.append("crawl_depth = ?"); params.append(crawl_depth)
        if url_pattern is not None:
            updates.append("url_pattern = ?"); params.append(url_pattern)
        if url_exclude is not None:
            updates.append("url_exclude = ?"); params.append(url_exclude)
        params.append(source_id)
        conn.execute(f"UPDATE web_sources SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    finally:
        conn.close()


def delete_web_source(source_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM web_sources WHERE id = ?", (source_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ─────────────────────────────────────────────
# API Keys
# ─────────────────────────────────────────────

def create_api_key(name: str, provider: str = "anthropic") -> dict:
    conn = get_connection()
    try:
        key = f"{'sk-ant-' if provider == 'anthropic' else 'sk-'}" + secrets.token_hex(16)
        # This stores a reference key, not the actual provider key
        # The actual API key is stored in .env
        ref_key = "mekey_" + secrets.token_hex(16)
        cursor = conn.execute(
            "INSERT INTO api_keys (key, name, provider) VALUES (?, ?, ?)",
            (ref_key, name, provider),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "key": ref_key, "name": name, "provider": provider}
    finally:
        conn.close()


def get_api_keys() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_api_key(key_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


# ─────────────────────────────────────────────
# Distribution Log
# ─────────────────────────────────────────────

def log_distribution(client_id: int, expert_slug: str, action: str,
                     status: str = "pending", details: str = "") -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO distribution_log (client_id, expert_slug, action, status, details)
            VALUES (?, ?, ?, ?, ?)
        """, (client_id, expert_slug, action, status, details))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_distribution_log(client_id: int = None, limit: int = 50) -> list[dict]:
    conn = get_connection()
    if client_id:
        rows = conn.execute("""
            SELECT dl.*, c.name as client_name
            FROM distribution_log dl
            JOIN clients c ON dl.client_id = c.id
            WHERE dl.client_id = ?
            ORDER BY dl.created_at DESC LIMIT ?
        """, (client_id, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT dl.*, c.name as client_name
            FROM distribution_log dl
            JOIN clients c ON dl.client_id = c.id
            ORDER BY dl.created_at DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────

def get_setting(key: str, default: str = "") -> str:
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO settings (key, value, updated_at) VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = datetime('now')
    """, (key, value, value))
    conn.commit()
    conn.close()


def get_all_settings() -> dict:
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM settings").fetchall()
    conn.close()
    return {r["key"]: r["value"] for r in rows}
