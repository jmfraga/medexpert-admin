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

        CREATE TABLE IF NOT EXISTS glossary_terms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            expert_id INTEGER NOT NULL,
            term TEXT NOT NULL,
            category TEXT DEFAULT '',
            synonyms TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (expert_id) REFERENCES experts(id) ON DELETE CASCADE,
            UNIQUE(expert_id, term)
        );

        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER,
            telegram_id INTEGER,
            expert_slug TEXT DEFAULT '',
            ticket_type TEXT NOT NULL CHECK(ticket_type IN ('transcription', 'bug', 'feature', 'support')),
            status TEXT DEFAULT 'open' CHECK(status IN ('open', 'in_progress', 'resolved', 'rejected')),
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            original_text TEXT DEFAULT '',
            suggested_text TEXT DEFAULT '',
            admin_notes TEXT DEFAULT '',
            admin_response TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            resolved_at TEXT,
            FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE SET NULL
        );

        -- Pricing Plans
        CREATE TABLE IF NOT EXISTS pricing_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_key TEXT UNIQUE NOT NULL,
            label TEXT NOT NULL,
            tier TEXT NOT NULL CHECK(tier IN ('basic', 'premium')),
            period TEXT NOT NULL CHECK(period IN ('monthly', 'annual')),
            usd_price REAL NOT NULL,
            mxn_price REAL NOT NULL,
            stripe_price_id TEXT DEFAULT '',
            paypal_plan_id TEXT DEFAULT '',
            mp_preapproval_id TEXT DEFAULT '',
            clip_plan_id TEXT DEFAULT '',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        -- Promotions
        CREATE TABLE IF NOT EXISTS promotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code TEXT UNIQUE NOT NULL,
            description TEXT DEFAULT '',
            discount_percent INTEGER NOT NULL DEFAULT 0,
            discount_amount_usd REAL DEFAULT 0,
            valid_from TEXT DEFAULT (datetime('now')),
            valid_until TEXT,
            max_uses INTEGER DEFAULT 0,
            used_count INTEGER DEFAULT 0,
            applies_to TEXT DEFAULT 'all',
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Verification Documents
        CREATE TABLE IF NOT EXISTS verification_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL,
            doc_type TEXT NOT NULL CHECK(doc_type IN ('cedula', 'ine')),
            file_path TEXT NOT NULL,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
            admin_notes TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            reviewed_at TEXT,
            FOREIGN KEY (telegram_id) REFERENCES bot_users(telegram_id) ON DELETE CASCADE
        );

        -- Referral Rewards
        CREATE TABLE IF NOT EXISTS referral_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            referrer_id INTEGER NOT NULL,
            referred_id INTEGER NOT NULL,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'earned', 'claimed', 'expired')),
            bonus_type TEXT DEFAULT 'free_month',
            bonus_value REAL DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            claimed_at TEXT,
            FOREIGN KEY (referrer_id) REFERENCES bot_users(telegram_id),
            FOREIGN KEY (referred_id) REFERENCES bot_users(telegram_id)
        );

        -- Broadcast Messages
        CREATE TABLE IF NOT EXISTS broadcast_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            message TEXT NOT NULL,
            target TEXT DEFAULT 'all' CHECK(target IN ('all', 'subscribers', 'premium', 'verified')),
            sent_count INTEGER DEFAULT 0,
            failed_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'draft' CHECK(status IN ('draft', 'sending', 'sent', 'failed')),
            created_at TEXT DEFAULT (datetime('now')),
            sent_at TEXT
        );

        -- Congress Events Calendar
        CREATE TABLE IF NOT EXISTS congress_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            short_name TEXT NOT NULL,
            society TEXT DEFAULT '',
            location TEXT DEFAULT '',
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            description TEXT DEFAULT '',
            url TEXT DEFAULT '',
            alert_days_before INTEGER DEFAULT 7,
            alert_sent INTEGER DEFAULT 0,
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now'))
        );

        -- Telegram Bot Users
        CREATE TABLE IF NOT EXISTS bot_users (
            telegram_id INTEGER PRIMARY KEY,
            username TEXT DEFAULT '',
            first_name TEXT DEFAULT '',
            last_name TEXT DEFAULT '',
            specialty TEXT DEFAULT '',
            is_verified INTEGER DEFAULT 0,
            referral_code TEXT UNIQUE,
            referred_by INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            last_activity TEXT
        );

        -- Telegram Bot Consultations
        CREATE TABLE IF NOT EXISTS bot_consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER NOT NULL,
            specialty TEXT NOT NULL,
            query_type TEXT NOT NULL CHECK(query_type IN ('voice', 'text')),
            query_text TEXT DEFAULT '',
            response_text TEXT DEFAULT '',
            response_time_seconds REAL DEFAULT 0,
            llm_provider TEXT DEFAULT '',
            llm_model TEXT DEFAULT '',
            tokens_input INTEGER DEFAULT 0,
            tokens_output INTEGER DEFAULT 0,
            rag_chunks_used INTEGER DEFAULT 0,
            is_free_tier INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (telegram_id) REFERENCES bot_users(telegram_id) ON DELETE CASCADE
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
    try:
        conn.execute("ALTER TABLE bot_consultations ADD COLUMN citations_json TEXT DEFAULT '[]'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE bot_consultations ADD COLUMN is_deepening INTEGER DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE bot_consultations ADD COLUMN parent_consultation_id INTEGER DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE bot_consultations ADD COLUMN user_feedback TEXT DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        conn.execute("ALTER TABLE bot_consultations ADD COLUMN clinical_metadata_json TEXT DEFAULT '{}'")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # Column already exists
    # Bot user extra fields
    for col, defn in [
        ("email", "TEXT DEFAULT NULL"),
        ("subscription_plan", "TEXT DEFAULT 'free'"),
        ("subscription_status", "TEXT DEFAULT NULL"),
        ("stripe_customer_id", "TEXT DEFAULT NULL"),
        ("subscription_started_at", "TEXT DEFAULT NULL"),
        ("subscription_expires_at", "TEXT DEFAULT NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE bot_users ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    try:
        conn.execute("ALTER TABLE glossary_terms ADD COLUMN synonyms TEXT DEFAULT ''")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    for col, defn in [
        ("telegram_id", "INTEGER"),
        ("admin_response", "TEXT DEFAULT ''"),
    ]:
        try:
            conn.execute(f"ALTER TABLE tickets ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    # Migrate tickets table to add 'support' to CHECK constraint
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='tickets'").fetchone()
        if row and "'support'" not in row[0]:
            conn.execute("ALTER TABLE tickets RENAME TO tickets_old")
            conn.execute("""
                CREATE TABLE tickets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    client_id INTEGER,
                    telegram_id INTEGER,
                    expert_slug TEXT DEFAULT '',
                    ticket_type TEXT NOT NULL CHECK(ticket_type IN ('transcription', 'bug', 'feature', 'support')),
                    status TEXT DEFAULT 'open' CHECK(status IN ('open', 'in_progress', 'resolved', 'rejected')),
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    original_text TEXT DEFAULT '',
                    suggested_text TEXT DEFAULT '',
                    admin_notes TEXT DEFAULT '',
                    admin_response TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now')),
                    resolved_at TEXT,
                    FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE SET NULL
                )
            """)
            conn.execute("""
                INSERT INTO tickets (id, client_id, expert_slug, ticket_type, status,
                    title, description, original_text, suggested_text, admin_notes,
                    created_at, resolved_at)
                SELECT id, client_id, expert_slug, ticket_type, status,
                    title, description, original_text, suggested_text, admin_notes,
                    created_at, resolved_at
                FROM tickets_old
            """)
            conn.execute("DROP TABLE tickets_old")
            conn.commit()
            console.print("[green]Tickets table migrated (added support type)[/green]")
    except Exception as e:
        console.print(f"[yellow]Tickets migration note: {e}[/yellow]")
    # Bot user verification columns
    for col, defn in [
        ("verification_status", "TEXT DEFAULT 'none'"),
        ("verification_notes", "TEXT DEFAULT ''"),
        ("verified_at", "TEXT DEFAULT NULL"),
        ("source_preferences_json", "TEXT DEFAULT NULL"),
        ("applied_promo_id", "INTEGER DEFAULT NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE bot_users ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    # Terms acceptance
    try:
        conn.execute("ALTER TABLE bot_users ADD COLUMN terms_accepted_at TEXT DEFAULT NULL")
        conn.commit()
    except sqlite3.OperationalError:
        pass
    # Migrate verification_documents to accept 'titulo' doc_type
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='verification_documents'").fetchone()
        if row and "'titulo'" not in row[0]:
            conn.execute("ALTER TABLE verification_documents RENAME TO verification_documents_old")
            conn.execute("""
                CREATE TABLE verification_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    telegram_id INTEGER NOT NULL,
                    doc_type TEXT NOT NULL CHECK(doc_type IN ('cedula', 'titulo', 'ine')),
                    file_path TEXT NOT NULL,
                    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
                    admin_notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now')),
                    reviewed_at TEXT,
                    FOREIGN KEY (telegram_id) REFERENCES bot_users(telegram_id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                INSERT INTO verification_documents
                SELECT * FROM verification_documents_old
            """)
            conn.execute("DROP TABLE verification_documents_old")
            conn.commit()
            console.print("[green]verification_documents migrated (added titulo doc_type)[/green]")
    except Exception as e:
        console.print(f"[yellow]Verification docs migration note: {e}[/yellow]")
    # Per-expert LLM config columns
    for col, defn in [
        ("base_provider", "TEXT DEFAULT NULL"),
        ("base_model", "TEXT DEFAULT NULL"),
        ("deepen_provider", "TEXT DEFAULT NULL"),
        ("deepen_model", "TEXT DEFAULT NULL"),
        ("deepen_premium_provider", "TEXT DEFAULT NULL"),
        ("deepen_premium_model", "TEXT DEFAULT NULL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE experts ADD COLUMN {col} {defn}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    # Seed default pricing plans if empty
    try:
        count = conn.execute("SELECT COUNT(*) as cnt FROM pricing_plans").fetchone()["cnt"]
        if count == 0:
            conn.executemany("""
                INSERT INTO pricing_plans (plan_key, label, tier, period, usd_price, mxn_price)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                ("basic_monthly", "Básico Mensual", "basic", "monthly", 14.99, 299),
                ("basic_annual", "Básico Anual", "basic", "annual", 149.90, 2990),
                ("premium_monthly", "Premium Mensual", "premium", "monthly", 24.99, 499),
                ("premium_annual", "Premium Anual", "premium", "annual", 249.90, 4990),
            ])
            conn.commit()
            console.print("[green]Default pricing plans seeded[/green]")
    except Exception:
        pass
    # Seed congress events if empty
    try:
        count = conn.execute("SELECT COUNT(*) as cnt FROM congress_events").fetchone()["cnt"]
        if count == 0:
            conn.executemany("""
                INSERT INTO congress_events (name, short_name, society, location, start_date, end_date, description, url, alert_days_before)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                ("ASCO Annual Meeting 2026", "ASCO 2026", "ASCO", "Chicago, IL, USA", "2026-05-29", "2026-06-02",
                 "Congreso anual de la American Society of Clinical Oncology", "https://meetings.asco.org/am", 14),
                ("ESMO Congress 2026", "ESMO 2026", "ESMO", "Roma, Italia", "2026-09-18", "2026-09-22",
                 "Congreso anual de la European Society for Medical Oncology", "https://www.esmo.org/meeting-calendar", 14),
                ("SMEO Congreso Nacional 2026", "SMEO 2026", "SMEO", "Ciudad de Mexico", "2026-10-14", "2026-10-17",
                 "Congreso nacional de la Sociedad Mexicana de Oncologia", "https://smeo.org.mx", 14),
                ("San Antonio Breast Cancer Symposium 2026", "SABCS 2026", "SABCS", "San Antonio, TX, USA", "2026-12-08", "2026-12-12",
                 "Simposio de cancer de mama", "https://www.sabcs.org", 10),
                ("ASCO GI 2027", "ASCO GI 2027", "ASCO", "San Francisco, CA, USA", "2027-01-22", "2027-01-24",
                 "ASCO Gastrointestinal Cancers Symposium", "https://meetings.asco.org/gi", 10),
            ])
            conn.commit()
            console.print("[green]Congress events seeded (5 events)[/green]")
    except Exception:
        pass
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
                  icon: str = None, base_provider: str = None,
                  base_model: str = None, deepen_provider: str = None,
                  deepen_model: str = None, deepen_premium_provider: str = None,
                  deepen_premium_model: str = None):
    conn = get_connection()
    updates, params = [], []
    if name is not None:
        updates.append("name = ?"); params.append(name)
    if system_prompt is not None:
        updates.append("system_prompt = ?"); params.append(system_prompt)
    if icon is not None:
        updates.append("icon = ?"); params.append(icon)
    # LLM config: empty string → NULL (use global default)
    for col, val in [("base_provider", base_provider), ("base_model", base_model),
                     ("deepen_provider", deepen_provider), ("deepen_model", deepen_model),
                     ("deepen_premium_provider", deepen_premium_provider),
                     ("deepen_premium_model", deepen_premium_model)]:
        if val is not None:
            updates.append(f"{col} = ?"); params.append(val if val else None)
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


def get_expert_llm_config(expert_slug: str) -> dict:
    """Get resolved LLM config for an expert (expert override or global fallback).

    Returns dict with: base_provider, base_model,
      deepen_provider, deepen_model (basic tier),
      deepen_premium_provider, deepen_premium_model (premium tier)
    """
    settings = get_all_settings()
    expert = get_expert_by_slug(expert_slug)

    base_provider = (expert.get("base_provider") if expert else None) or settings.get("default_provider", "anthropic")
    base_model = (expert.get("base_model") if expert else None) or settings.get("default_model", "claude-haiku-4-5-20251001")
    deepen_provider = (expert.get("deepen_provider") if expert else None) or settings.get("default_deepen_provider", "synapse")
    deepen_model = (expert.get("deepen_model") if expert else None) or settings.get("default_deepen_model", "auto")
    deepen_premium_provider = (expert.get("deepen_premium_provider") if expert else None) or settings.get("default_deepen_premium_provider", "synapse")
    deepen_premium_model = (expert.get("deepen_premium_model") if expert else None) or settings.get("default_deepen_premium_model", "auto")

    return {
        "base_provider": base_provider,
        "base_model": base_model,
        "deepen_provider": deepen_provider,
        "deepen_model": deepen_model,
        "deepen_premium_provider": deepen_premium_provider,
        "deepen_premium_model": deepen_premium_model,
    }


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
# Glossary Terms
# ─────────────────────────────────────────────

def create_glossary_term(expert_id: int, term: str, category: str = "", synonyms: str = "") -> int:
    conn = get_connection()
    try:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO glossary_terms (expert_id, term, category, synonyms) VALUES (?, ?, ?, ?)",
            (expert_id, term, category, synonyms),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_glossary_terms_for_expert(expert_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM glossary_terms WHERE expert_id = ? ORDER BY category, term",
        (expert_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_glossary_term(term_id: int, **kwargs):
    conn = get_connection()
    allowed = {"term", "category", "synonyms"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            updates.append(f"{key} = ?"); params.append(val)
    if updates:
        params.append(term_id)
        conn.execute(f"UPDATE glossary_terms SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()


def delete_glossary_term(term_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM glossary_terms WHERE id = ?", (term_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def bulk_create_glossary_terms(expert_id: int, terms: list[dict]) -> int:
    conn = get_connection()
    try:
        count = 0
        for t in terms:
            term = t.get("term", "").strip()
            if not term:
                continue
            category = t.get("category", "").strip()
            synonyms = t.get("synonyms", "").strip()
            cursor = conn.execute(
                "INSERT OR IGNORE INTO glossary_terms (expert_id, term, category, synonyms) VALUES (?, ?, ?, ?)",
                (expert_id, term, category, synonyms),
            )
            count += cursor.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def get_glossary_terms_for_expert_by_slug(expert_slug: str) -> list[dict]:
    """Get glossary terms by expert slug (used by bot for Whisper prompt)."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT gt.* FROM glossary_terms gt
        JOIN experts e ON gt.expert_id = e.id
        WHERE e.slug = ? ORDER BY gt.term
    """, (expert_slug,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_glossary_term_count(expert_id: int) -> int:
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM glossary_terms WHERE expert_id = ?",
        (expert_id,)
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


# ─────────────────────────────────────────────
# Tickets
# ─────────────────────────────────────────────

def create_ticket(client_id: int = None, ticket_type: str = "support", title: str = "",
                  description: str = "", expert_slug: str = "",
                  original_text: str = "", suggested_text: str = "",
                  telegram_id: int = None) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO tickets (client_id, telegram_id, ticket_type, title, description,
                                 expert_slug, original_text, suggested_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (client_id, telegram_id, ticket_type, title, description,
              expert_slug, original_text, suggested_text))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_tickets(status: str = None, ticket_type: str = None) -> list[dict]:
    conn = get_connection()
    query = "SELECT * FROM tickets"
    conditions, params = [], []
    if status:
        conditions.append("status = ?"); params.append(status)
    if ticket_type:
        conditions.append("ticket_type = ?"); params.append(ticket_type)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC"
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_ticket_by_id(ticket_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_ticket(ticket_id: int, **kwargs):
    conn = get_connection()
    allowed = {"status", "admin_notes", "admin_response", "resolved_at"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            updates.append(f"{key} = ?"); params.append(val)
    if updates:
        params.append(ticket_id)
        conn.execute(f"UPDATE tickets SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()


def get_ticket_stats() -> dict:
    conn = get_connection()
    total = conn.execute("SELECT COUNT(*) as cnt FROM tickets").fetchone()["cnt"]
    open_count = conn.execute("SELECT COUNT(*) as cnt FROM tickets WHERE status = 'open'").fetchone()["cnt"]
    in_progress = conn.execute("SELECT COUNT(*) as cnt FROM tickets WHERE status = 'in_progress'").fetchone()["cnt"]
    resolved = conn.execute("SELECT COUNT(*) as cnt FROM tickets WHERE status = 'resolved'").fetchone()["cnt"]
    transcription = conn.execute("SELECT COUNT(*) as cnt FROM tickets WHERE ticket_type = 'transcription'").fetchone()["cnt"]
    bug = conn.execute("SELECT COUNT(*) as cnt FROM tickets WHERE ticket_type = 'bug'").fetchone()["cnt"]
    feature = conn.execute("SELECT COUNT(*) as cnt FROM tickets WHERE ticket_type = 'feature'").fetchone()["cnt"]
    conn.close()
    return {
        "total": total,
        "open": open_count,
        "in_progress": in_progress,
        "resolved": resolved,
        "by_type": {
            "transcription": transcription,
            "bug": bug,
            "feature": feature,
        },
    }


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


# ─────────────────────────────────────────────
# Telegram Bot Users
# ─────────────────────────────────────────────

def get_or_create_bot_user(telegram_id: int, username: str = "",
                           first_name: str = "", last_name: str = "",
                           specialty: str = "") -> dict:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM bot_users WHERE telegram_id = ?",
                           (telegram_id,)).fetchone()
        if row:
            # Update last_activity and username if changed
            conn.execute("""
                UPDATE bot_users SET last_activity = datetime('now'),
                    username = COALESCE(NULLIF(?, ''), username),
                    first_name = COALESCE(NULLIF(?, ''), first_name)
                WHERE telegram_id = ?
            """, (username, first_name, telegram_id))
            conn.commit()
            return dict(conn.execute("SELECT * FROM bot_users WHERE telegram_id = ?",
                                     (telegram_id,)).fetchone())
        # Create new user with referral code
        ref_code = f"{specialty[:4].upper()}-{secrets.token_hex(3).upper()}" if specialty else f"ME-{secrets.token_hex(3).upper()}"
        conn.execute("""
            INSERT INTO bot_users (telegram_id, username, first_name, last_name,
                                   specialty, referral_code, last_activity)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """, (telegram_id, username, first_name, last_name, specialty, ref_code))
        conn.commit()
        return dict(conn.execute("SELECT * FROM bot_users WHERE telegram_id = ?",
                                 (telegram_id,)).fetchone())
    finally:
        conn.close()


def get_bot_user(telegram_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM bot_users WHERE telegram_id = ?",
                       (telegram_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_bot_user(telegram_id: int, **kwargs):
    conn = get_connection()
    allowed = {"username", "first_name", "last_name", "specialty",
               "is_verified", "referred_by", "last_activity", "email",
               "source_preferences_json", "terms_accepted_at"}
    nullable = {"source_preferences_json"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and (val is not None or key in nullable):
            updates.append(f"{key} = ?"); params.append(val)
    if updates:
        params.append(telegram_id)
        conn.execute(f"UPDATE bot_users SET {', '.join(updates)} WHERE telegram_id = ?", params)
        conn.commit()
    conn.close()


def get_bot_user_sources(telegram_id: int) -> list[str] | None:
    """Return user's enabled source societies, or None if all enabled (default)."""
    user = get_bot_user(telegram_id)
    if not user or not user.get("source_preferences_json"):
        return None
    try:
        sources = json.loads(user["source_preferences_json"])
        return sources if isinstance(sources, list) and sources else None
    except (json.JSONDecodeError, TypeError):
        return None


def count_bot_free_queries(telegram_id: int, specialty: str) -> int:
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM bot_consultations
        WHERE telegram_id = ? AND specialty = ? AND is_free_tier = 1
    """, (telegram_id, specialty)).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def can_bot_user_query(telegram_id: int, specialty: str, free_limit: int = 5) -> bool:
    """Check if user can make a query (free tier or subscribed)."""
    user = get_bot_user(telegram_id)
    if user and user.get("subscription_plan") in ("basic", "premium"):
        if user.get("subscription_status") == "active":
            return True
    used = count_bot_free_queries(telegram_id, specialty)
    return used < free_limit


def count_bot_paid_queries(telegram_id: int) -> int:
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM bot_consultations
        WHERE telegram_id = ? AND is_free_tier = 0
    """, (telegram_id,)).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def get_bot_user_plan(telegram_id: int) -> str:
    """Return user's active plan: 'free', 'basic', or 'premium'."""
    user = get_bot_user(telegram_id)
    if not user:
        return "free"
    if user.get("subscription_plan") in ("basic", "premium"):
        if user.get("subscription_status") == "active":
            return user["subscription_plan"]
    return "free"


def update_bot_user_subscription(telegram_id: int, plan: str, status: str,
                                  stripe_customer_id: str = None) -> bool:
    """Update a user's subscription after Stripe payment."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE bot_users
            SET subscription_plan = ?,
                subscription_status = ?,
                stripe_customer_id = COALESCE(?, stripe_customer_id),
                subscription_started_at = datetime('now'),
                subscription_expires_at = datetime('now', '+30 days')
            WHERE telegram_id = ?
        """, (plan, status, stripe_customer_id, telegram_id))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def cancel_bot_user_subscription(telegram_id: int) -> bool:
    """Cancel a user's subscription."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE bot_users
            SET subscription_status = 'cancelled'
            WHERE telegram_id = ?
        """, (telegram_id,))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Telegram Bot Consultations
# ─────────────────────────────────────────────

def count_bot_deepenings_month(telegram_id: int, specialty: str) -> int:
    """Count deepening queries this month for a user."""
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM bot_consultations
        WHERE telegram_id = ? AND specialty = ? AND is_deepening = 1
        AND created_at >= date('now', 'start of month')
    """, (telegram_id, specialty)).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def count_bot_opus_deepenings_today(telegram_id: int, specialty: str) -> int:
    """Count Opus (premium) deepenings today for a user."""
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM bot_consultations
        WHERE telegram_id = ? AND specialty = ? AND is_deepening = 1
        AND llm_model = 'claude-opus-4-6'
        AND date(created_at) = date('now')
    """, (telegram_id, specialty)).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def count_bot_sonnet_deepenings_today(telegram_id: int, specialty: str) -> int:
    """Count Sonnet (basic) deepenings today for a user."""
    conn = get_connection()
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM bot_consultations
        WHERE telegram_id = ? AND specialty = ? AND is_deepening = 1
        AND llm_model = 'claude-sonnet-4-20250514'
        AND date(created_at) = date('now')
    """, (telegram_id, specialty)).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def log_bot_consultation(telegram_id: int, specialty: str, query_type: str,
                         query_text: str = "", response_text: str = "",
                         response_time_seconds: float = 0,
                         llm_provider: str = "", llm_model: str = "",
                         tokens_input: int = 0, tokens_output: int = 0,
                         rag_chunks_used: int = 0, is_free_tier: bool = True,
                         citations: list[str] | None = None,
                         is_deepening: bool = False,
                         parent_consultation_id: int | None = None,
                         clinical_metadata_json: str = "{}") -> int:
    import json
    citations_json = json.dumps(citations or [])
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO bot_consultations
                (telegram_id, specialty, query_type, query_text, response_text,
                 response_time_seconds, llm_provider, llm_model,
                 tokens_input, tokens_output, rag_chunks_used, is_free_tier,
                 citations_json, is_deepening, parent_consultation_id,
                 clinical_metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (telegram_id, specialty, query_type, query_text, response_text,
              response_time_seconds, llm_provider, llm_model,
              tokens_input, tokens_output, rag_chunks_used,
              1 if is_free_tier else 0, citations_json,
              1 if is_deepening else 0, parent_consultation_id,
              clinical_metadata_json))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_bot_stats() -> dict:
    """Get bot usage statistics for admin dashboard."""
    conn = get_connection()
    try:
        total_users = conn.execute("SELECT COUNT(*) as cnt FROM bot_users").fetchone()["cnt"]
        verified_users = conn.execute("SELECT COUNT(*) as cnt FROM bot_users WHERE is_verified = 1").fetchone()["cnt"]
        total_queries = conn.execute("SELECT COUNT(*) as cnt FROM bot_consultations").fetchone()["cnt"]
        voice_queries = conn.execute("SELECT COUNT(*) as cnt FROM bot_consultations WHERE query_type = 'voice'").fetchone()["cnt"]
        text_queries = conn.execute("SELECT COUNT(*) as cnt FROM bot_consultations WHERE query_type = 'text'").fetchone()["cnt"]
        today_queries = conn.execute("""
            SELECT COUNT(*) as cnt FROM bot_consultations
            WHERE date(created_at) = date('now')
        """).fetchone()["cnt"]
        # Users active in last 7 days
        active_7d = conn.execute("""
            SELECT COUNT(*) as cnt FROM bot_users
            WHERE last_activity >= datetime('now', '-7 days')
        """).fetchone()["cnt"]
        paid_users = conn.execute(
            "SELECT COUNT(*) as cnt FROM bot_users WHERE subscription_status = 'active'"
        ).fetchone()["cnt"]
        deepenings = conn.execute(
            "SELECT COUNT(*) as cnt FROM bot_consultations WHERE is_deepening = 1"
        ).fetchone()["cnt"]
        feedback_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM bot_consultations WHERE user_feedback IS NOT NULL"
        ).fetchone()["cnt"]
        return {
            "total_users": total_users,
            "verified_users": verified_users,
            "active_7d": active_7d,
            "paid_users": paid_users,
            "total_queries": total_queries,
            "voice_queries": voice_queries,
            "text_queries": text_queries,
            "today_queries": today_queries,
            "deepenings": deepenings,
            "feedback_count": feedback_count,
        }
    finally:
        conn.close()


def get_analytics_data(days: int | None = 30) -> dict:
    """Get comprehensive analytics data for the analytics dashboard.

    Args:
        days: Number of days to look back (7, 30, 90, or None for all time)

    Returns:
        Dict with kpis, time_series, distributions, clinical, and feedback sections
    """
    import json as _json
    conn = get_connection()
    try:
        date_filter = ""
        date_filter_users = ""
        if days:
            date_filter = f"AND bc.created_at >= datetime('now', '-{days} days')"
            date_filter_users = f"AND bu.created_at >= datetime('now', '-{days} days')"

        # ── KPIs ──
        total_queries = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM bot_consultations bc WHERE 1=1 {date_filter}
        """).fetchone()["cnt"]

        today_queries = conn.execute("""
            SELECT COUNT(*) as cnt FROM bot_consultations
            WHERE date(created_at) = date('now')
        """).fetchone()["cnt"]

        avg_response = conn.execute(f"""
            SELECT COALESCE(AVG(bc.response_time_seconds), 0) as avg_rt
            FROM bot_consultations bc WHERE bc.response_time_seconds > 0 {date_filter}
        """).fetchone()["avg_rt"]

        active_7d = conn.execute("""
            SELECT COUNT(*) as cnt FROM bot_users
            WHERE last_activity >= datetime('now', '-7 days')
        """).fetchone()["cnt"]

        active_30d = conn.execute("""
            SELECT COUNT(*) as cnt FROM bot_users
            WHERE last_activity >= datetime('now', '-30 days')
        """).fetchone()["cnt"]

        total_users = conn.execute("SELECT COUNT(*) as cnt FROM bot_users").fetchone()["cnt"]
        paid_users = conn.execute(
            "SELECT COUNT(*) as cnt FROM bot_users WHERE subscription_status = 'active'"
        ).fetchone()["cnt"]
        conversion_rate = round((paid_users / total_users * 100) if total_users > 0 else 0, 1)

        top_model_row = conn.execute(f"""
            SELECT bc.llm_model, COUNT(*) as cnt FROM bot_consultations bc
            WHERE bc.llm_model != '' {date_filter}
            GROUP BY bc.llm_model ORDER BY cnt DESC LIMIT 1
        """).fetchone()
        top_model = top_model_row["llm_model"] if top_model_row else "N/A"

        feedback_rows = conn.execute(f"""
            SELECT bc.user_feedback, COUNT(*) as cnt FROM bot_consultations bc
            WHERE bc.user_feedback IS NOT NULL {date_filter}
            GROUP BY bc.user_feedback
        """).fetchall()
        feedback_total = sum(r["cnt"] for r in feedback_rows)
        feedback_dist = {r["user_feedback"]: r["cnt"] for r in feedback_rows}
        # Positive: "Mejoró mi manera de ver las cosas", "Reforzó mi plan"
        _POSITIVE = {"Mejoró mi manera de ver las cosas", "Reforzó mi plan"}
        positive_fb = sum(v for k, v in feedback_dist.items() if k in _POSITIVE)
        feedback_score = round((positive_fb / feedback_total * 100) if feedback_total > 0 else 0, 1)

        kpis = {
            "total_queries": total_queries,
            "today_queries": today_queries,
            "avg_response_time": round(avg_response, 1),
            "active_users_7d": active_7d,
            "active_users_30d": active_30d,
            "total_users": total_users,
            "paid_users": paid_users,
            "conversion_rate": conversion_rate,
            "top_model": top_model,
            "feedback_score": feedback_score,
            "feedback_total": feedback_total,
        }

        # ── Time Series ──
        queries_per_day = [dict(r) for r in conn.execute(f"""
            SELECT date(bc.created_at) as day, COUNT(*) as count
            FROM bot_consultations bc WHERE 1=1 {date_filter}
            GROUP BY day ORDER BY day
        """).fetchall()]

        registrations_per_day = [dict(r) for r in conn.execute(f"""
            SELECT date(bu.created_at) as day, COUNT(*) as count
            FROM bot_users bu WHERE 1=1 {date_filter_users}
            GROUP BY day ORDER BY day
        """).fetchall()]

        response_time_trend = [dict(r) for r in conn.execute(f"""
            SELECT date(bc.created_at) as day,
                   ROUND(AVG(bc.response_time_seconds), 1) as avg_rt
            FROM bot_consultations bc
            WHERE bc.response_time_seconds > 0 {date_filter}
            GROUP BY day ORDER BY day
        """).fetchall()]

        tokens_per_day = [dict(r) for r in conn.execute(f"""
            SELECT date(bc.created_at) as day,
                   SUM(bc.tokens_input) as tokens_in,
                   SUM(bc.tokens_output) as tokens_out
            FROM bot_consultations bc WHERE 1=1 {date_filter}
            GROUP BY day ORDER BY day
        """).fetchall()]

        time_series = {
            "queries_per_day": queries_per_day,
            "registrations_per_day": registrations_per_day,
            "response_time_trend": response_time_trend,
            "tokens_per_day": tokens_per_day,
        }

        # ── Distributions ──
        voice_count = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM bot_consultations bc
            WHERE bc.query_type = 'voice' {date_filter}
        """).fetchone()["cnt"]
        text_count = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM bot_consultations bc
            WHERE bc.query_type = 'text' {date_filter}
        """).fetchone()["cnt"]

        busiest_hours = [dict(r) for r in conn.execute(f"""
            SELECT CAST(strftime('%H', bc.created_at) AS INTEGER) as hour, COUNT(*) as count
            FROM bot_consultations bc WHERE 1=1 {date_filter}
            GROUP BY hour ORDER BY hour
        """).fetchall()]

        # Guideline society usage from citations_json
        all_citations = conn.execute(f"""
            SELECT bc.citations_json FROM bot_consultations bc
            WHERE bc.citations_json != '[]' AND bc.citations_json IS NOT NULL {date_filter}
        """).fetchall()
        society_counts = {"NCCN": 0, "ESMO": 0, "IMSS": 0, "NCI": 0, "Otro": 0}
        for row in all_citations:
            try:
                cites = _json.loads(row["citations_json"])
                for cite in cites:
                    cite_upper = cite.upper() if isinstance(cite, str) else str(cite).upper()
                    if "NCCN" in cite_upper:
                        society_counts["NCCN"] += 1
                    elif "ESMO" in cite_upper:
                        society_counts["ESMO"] += 1
                    elif "IMSS" in cite_upper or "GPC" in cite_upper:
                        society_counts["IMSS"] += 1
                    elif "NCI" in cite_upper or "PDQ" in cite_upper:
                        society_counts["NCI"] += 1
                    else:
                        society_counts["Otro"] += 1
            except (_json.JSONDecodeError, TypeError):
                pass

        model_dist = [dict(r) for r in conn.execute(f"""
            SELECT bc.llm_model as model, COUNT(*) as count
            FROM bot_consultations bc
            WHERE bc.llm_model != '' {date_filter}
            GROUP BY bc.llm_model ORDER BY count DESC
        """).fetchall()]

        distributions = {
            "voice_vs_text": {"voice": voice_count, "text": text_count},
            "busiest_hours": busiest_hours,
            "guideline_societies": society_counts,
            "model_distribution": model_dist,
        }

        # ── Clinical Stats ──
        specialty_dist = [dict(r) for r in conn.execute(f"""
            SELECT bc.specialty, COUNT(*) as count
            FROM bot_consultations bc WHERE 1=1 {date_filter}
            GROUP BY bc.specialty ORDER BY count DESC
        """).fetchall()]

        deepening_count = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM bot_consultations bc
            WHERE bc.is_deepening = 1 {date_filter}
        """).fetchone()["cnt"]
        deepening_rate = round((deepening_count / total_queries * 100) if total_queries > 0 else 0, 1)

        rag_usage = conn.execute(f"""
            SELECT COALESCE(AVG(bc.rag_chunks_used), 0) as avg_chunks,
                   SUM(CASE WHEN bc.rag_chunks_used > 0 THEN 1 ELSE 0 END) as with_rag,
                   COUNT(*) as total
            FROM bot_consultations bc WHERE 1=1 {date_filter}
        """).fetchone()
        rag_coverage = round((rag_usage["with_rag"] / rag_usage["total"] * 100)
                             if rag_usage["total"] > 0 else 0, 1)

        # Free vs paid query split
        free_queries = conn.execute(f"""
            SELECT COUNT(*) as cnt FROM bot_consultations bc
            WHERE bc.is_free_tier = 1 {date_filter}
        """).fetchone()["cnt"]
        paid_queries = total_queries - free_queries

        clinical = {
            "specialty_distribution": specialty_dist,
            "deepening_count": deepening_count,
            "deepening_rate": deepening_rate,
            "avg_rag_chunks": round(rag_usage["avg_chunks"], 1),
            "rag_coverage": rag_coverage,
            "free_queries": free_queries,
            "paid_queries": paid_queries,
        }

        # ── Feedback / Decision Support ──
        feedback_by_day = [dict(r) for r in conn.execute(f"""
            SELECT date(bc.created_at) as day,
                   SUM(CASE WHEN bc.user_feedback IN ('Mejoró mi manera de ver las cosas', 'Reforzó mi plan') THEN 1 ELSE 0 END) as positive,
                   SUM(CASE WHEN bc.user_feedback = 'Información incompleta' THEN 1 ELSE 0 END) as neutral,
                   SUM(CASE WHEN bc.user_feedback IN ('Información incorrecta', 'No me sirvió (otra razón)') THEN 1 ELSE 0 END) as negative
            FROM bot_consultations bc
            WHERE bc.user_feedback IS NOT NULL {date_filter}
            GROUP BY day ORDER BY day
        """).fetchall()]

        feedback_response_rate = round(
            (feedback_total / total_queries * 100) if total_queries > 0 else 0, 1
        )

        # Deepening trend per day
        deepening_per_day = [dict(r) for r in conn.execute(f"""
            SELECT date(bc.created_at) as day, COUNT(*) as count
            FROM bot_consultations bc
            WHERE bc.is_deepening = 1 {date_filter}
            GROUP BY day ORDER BY day
        """).fetchall()]

        feedback = {
            "distribution": feedback_dist,
            "total": feedback_total,
            "score": feedback_score,
            "response_rate": feedback_response_rate,
            "by_day": feedback_by_day,
            "deepening_per_day": deepening_per_day,
        }

        # ── API Costs (estimated) ──
        _COST_PER_1K = {
            # (input_per_1k, output_per_1k) in USD
            "claude-sonnet-4-20250514": (0.003, 0.015),
            "claude-opus-4-6": (0.015, 0.075),
            "openai/gpt-oss-120b": (0.0, 0.0),  # Groq free tier
        }

        cost_by_model = conn.execute(f"""
            SELECT bc.llm_model,
                   SUM(bc.tokens_input) as total_in,
                   SUM(bc.tokens_output) as total_out,
                   COUNT(*) as queries
            FROM bot_consultations bc
            WHERE bc.llm_model != '' {date_filter}
            GROUP BY bc.llm_model ORDER BY total_out DESC
        """).fetchall()

        api_costs = []
        total_cost = 0.0
        for row in cost_by_model:
            model = row["llm_model"]
            rates = _COST_PER_1K.get(model, (0.003, 0.015))  # default Sonnet rates
            cost_in = (row["total_in"] / 1000) * rates[0]
            cost_out = (row["total_out"] / 1000) * rates[1]
            cost = round(cost_in + cost_out, 4)
            total_cost += cost
            api_costs.append({
                "model": model,
                "tokens_input": row["total_in"],
                "tokens_output": row["total_out"],
                "queries": row["queries"],
                "cost_usd": cost,
            })

        cost_per_day = [dict(r) for r in conn.execute(f"""
            SELECT date(bc.created_at) as day,
                   bc.llm_model as model,
                   SUM(bc.tokens_input) as tokens_in,
                   SUM(bc.tokens_output) as tokens_out
            FROM bot_consultations bc
            WHERE bc.llm_model != '' {date_filter}
            GROUP BY day, bc.llm_model ORDER BY day
        """).fetchall()]

        # Compute daily costs
        daily_costs = {}
        for row in cost_per_day:
            day = row["day"]
            rates = _COST_PER_1K.get(row["model"], (0.003, 0.015))
            cost = (row["tokens_in"] / 1000) * rates[0] + (row["tokens_out"] / 1000) * rates[1]
            daily_costs[day] = round(daily_costs.get(day, 0) + cost, 4)
        cost_trend = [{"day": d, "cost": c} for d, c in sorted(daily_costs.items())]

        total_tokens = sum(r["tokens_input"] + r["tokens_output"] for r in api_costs)
        avg_cost_per_query = round(total_cost / total_queries, 4) if total_queries > 0 else 0

        api_usage = {
            "by_model": api_costs,
            "total_cost_usd": round(total_cost, 2),
            "avg_cost_per_query": avg_cost_per_query,
            "total_tokens": total_tokens,
            "cost_trend": cost_trend,
        }

        # ── Diagnosis-Deepening Correlation ──
        # Which specialties/queries get deepened most
        deepen_by_specialty = [dict(r) for r in conn.execute(f"""
            SELECT bc.specialty,
                   COUNT(*) as total,
                   SUM(CASE WHEN bc.is_deepening = 1 THEN 1 ELSE 0 END) as deepened
            FROM bot_consultations bc WHERE 1=1 {date_filter}
            GROUP BY bc.specialty ORDER BY total DESC
        """).fetchall()]
        for row in deepen_by_specialty:
            row["deepening_rate"] = round(
                (row["deepened"] / row["total"] * 100) if row["total"] > 0 else 0, 1
            )

        # Top queries that were deepened (query text snippets)
        top_deepened = [dict(r) for r in conn.execute(f"""
            SELECT bc.query_text, COUNT(*) as times_deepened
            FROM bot_consultations bc
            WHERE bc.is_deepening = 1 AND bc.query_text != '' {date_filter}
            GROUP BY bc.query_text ORDER BY times_deepened DESC LIMIT 10
        """).fetchall()]
        # Truncate query text for display
        for row in top_deepened:
            if len(row["query_text"]) > 80:
                row["query_text"] = row["query_text"][:80] + "..."

        clinical["deepen_by_specialty"] = deepen_by_specialty
        clinical["top_deepened_queries"] = top_deepened

        # ── Clinical Metadata Analytics ──
        # Top diagnoses (with CIE-10 code)
        meta_diagnoses = [dict(r) for r in conn.execute(f"""
            SELECT json_extract(bc.clinical_metadata_json, '$.diagnosis') as diagnosis,
                   MAX(json_extract(bc.clinical_metadata_json, '$.cie10')) as cie10,
                   COUNT(*) as count
            FROM bot_consultations bc
            WHERE json_extract(bc.clinical_metadata_json, '$.diagnosis') IS NOT NULL
              AND json_extract(bc.clinical_metadata_json, '$.diagnosis') != 'null'
              {date_filter}
            GROUP BY diagnosis ORDER BY count DESC LIMIT 15
        """).fetchall()]

        # Category distribution (solida vs hematologica)
        meta_categories = [dict(r) for r in conn.execute(f"""
            SELECT json_extract(bc.clinical_metadata_json, '$.category') as category,
                   COUNT(*) as count
            FROM bot_consultations bc
            WHERE json_extract(bc.clinical_metadata_json, '$.category') IS NOT NULL
              AND json_extract(bc.clinical_metadata_json, '$.category') != 'null'
              {date_filter}
            GROUP BY category ORDER BY count DESC
        """).fetchall()]

        # Stage distribution
        meta_stages = [dict(r) for r in conn.execute(f"""
            SELECT json_extract(bc.clinical_metadata_json, '$.clinical_details.stage') as stage,
                   COUNT(*) as count
            FROM bot_consultations bc
            WHERE json_extract(bc.clinical_metadata_json, '$.clinical_details.stage') IS NOT NULL
              AND json_extract(bc.clinical_metadata_json, '$.clinical_details.stage') != 'null'
              {date_filter}
            GROUP BY stage ORDER BY count DESC
        """).fetchall()]

        # Intent distribution
        meta_intents = [dict(r) for r in conn.execute(f"""
            SELECT json_extract(bc.clinical_metadata_json, '$.intent') as intent,
                   COUNT(*) as count
            FROM bot_consultations bc
            WHERE json_extract(bc.clinical_metadata_json, '$.intent') IS NOT NULL
              AND json_extract(bc.clinical_metadata_json, '$.intent') != 'null'
              {date_filter}
            GROUP BY intent ORDER BY count DESC
        """).fetchall()]

        # Top treatments (parse JSON array from each row)
        all_meta_rows = conn.execute(f"""
            SELECT bc.clinical_metadata_json FROM bot_consultations bc
            WHERE bc.clinical_metadata_json != '{{}}'
              AND bc.clinical_metadata_json IS NOT NULL
              {date_filter}
        """).fetchall()
        treatment_counts: dict[str, int] = {}
        meta_with_data = 0
        for row in all_meta_rows:
            try:
                meta = _json.loads(row["clinical_metadata_json"])
                if meta.get("diagnosis"):
                    meta_with_data += 1
                for drug in meta.get("treatments_mentioned", []):
                    treatment_counts[drug] = treatment_counts.get(drug, 0) + 1
            except (_json.JSONDecodeError, TypeError):
                pass
        meta_treatments = sorted(
            [{"treatment": k, "count": v} for k, v in treatment_counts.items()],
            key=lambda x: x["count"], reverse=True,
        )[:15]

        meta_coverage = round((meta_with_data / total_queries * 100) if total_queries > 0 else 0, 1)

        # Diagnosis × deepening rate (with CIE-10)
        meta_diag_deepen = [dict(r) for r in conn.execute(f"""
            SELECT json_extract(bc.clinical_metadata_json, '$.diagnosis') as diagnosis,
                   MAX(json_extract(bc.clinical_metadata_json, '$.cie10')) as cie10,
                   COUNT(*) as total,
                   SUM(CASE WHEN bc.is_deepening = 1 THEN 1 ELSE 0 END) as deepened
            FROM bot_consultations bc
            WHERE json_extract(bc.clinical_metadata_json, '$.diagnosis') IS NOT NULL
              AND json_extract(bc.clinical_metadata_json, '$.diagnosis') != 'null'
              {date_filter}
            GROUP BY diagnosis ORDER BY total DESC LIMIT 15
        """).fetchall()]
        for row in meta_diag_deepen:
            row["deepening_rate"] = round(
                (row["deepened"] / row["total"] * 100) if row["total"] > 0 else 0, 1
            )

        metadata = {
            "coverage_pct": meta_coverage,
            "top_diagnoses": meta_diagnoses,
            "category_distribution": meta_categories,
            "stage_distribution": meta_stages,
            "intent_distribution": meta_intents,
            "top_treatments": meta_treatments,
            "diagnosis_deepening": meta_diag_deepen,
        }

        return {
            "kpis": kpis,
            "time_series": time_series,
            "distributions": distributions,
            "clinical": clinical,
            "feedback": feedback,
            "api_usage": api_usage,
            "metadata": metadata,
        }
    finally:
        conn.close()


def get_analytics_export_data(days: int | None, section: str) -> dict:
    """Export analytics data as lists of dicts for CSV generation.

    Args:
        days: Number of days to look back (None for all time)
        section: 'consultations', 'users', 'costs', or 'all'

    Returns:
        Dict with keys matching sections, each value a list of dicts (rows).
    """
    import json as _json
    conn = get_connection()
    try:
        date_filter = ""
        date_filter_users = ""
        if days:
            date_filter = f"AND bc.created_at >= datetime('now', '-{days} days')"
            date_filter_users = f"AND bu.created_at >= datetime('now', '-{days} days')"

        result = {}

        if section in ("consultations", "all"):
            rows = conn.execute(f"""
                SELECT bc.id, bc.created_at, bc.telegram_id,
                       COALESCE(bu.username, '') as username,
                       bc.specialty, bc.query_type, bc.llm_model,
                       bc.tokens_input, bc.tokens_output,
                       ROUND(bc.response_time_seconds, 2) as response_time,
                       bc.is_deepening, bc.user_feedback,
                       bc.rag_chunks_used, bc.is_free_tier,
                       SUBSTR(bc.query_text, 1, 200) as query_text_short,
                       bc.clinical_metadata_json
                FROM bot_consultations bc
                LEFT JOIN bot_users bu ON bc.telegram_id = bu.telegram_id
                WHERE 1=1 {date_filter}
                ORDER BY bc.created_at DESC
            """).fetchall()
            result["consultations"] = [dict(r) for r in rows]

        if section in ("users", "all"):
            rows = conn.execute(f"""
                SELECT bu.telegram_id, bu.username, bu.first_name, bu.last_name,
                       bu.subscription_plan, bu.subscription_status,
                       bu.created_at, bu.last_activity, bu.email,
                       bu.is_verified, bu.specialty,
                       (SELECT COUNT(*) FROM bot_consultations bc
                        WHERE bc.telegram_id = bu.telegram_id) as total_queries
                FROM bot_users bu
                WHERE 1=1 {date_filter_users}
                ORDER BY bu.last_activity DESC
            """).fetchall()
            result["users"] = [dict(r) for r in rows]

        if section in ("costs", "all"):
            _COST_PER_1K = {
                "claude-sonnet-4-20250514": (0.003, 0.015),
                "claude-opus-4-6": (0.015, 0.075),
                "openai/gpt-oss-120b": (0.0, 0.0),
            }
            rows = conn.execute(f"""
                SELECT date(bc.created_at) as day, bc.llm_model as model,
                       SUM(bc.tokens_input) as tokens_in,
                       SUM(bc.tokens_output) as tokens_out,
                       COUNT(*) as queries
                FROM bot_consultations bc
                WHERE bc.llm_model != '' {date_filter}
                GROUP BY day, bc.llm_model ORDER BY day DESC
            """).fetchall()
            costs = []
            for row in rows:
                rates = _COST_PER_1K.get(row["model"], (0.003, 0.015))
                cost = round(
                    (row["tokens_in"] / 1000) * rates[0] + (row["tokens_out"] / 1000) * rates[1], 4
                )
                costs.append({
                    "day": row["day"],
                    "model": row["model"],
                    "queries": row["queries"],
                    "tokens_in": row["tokens_in"],
                    "tokens_out": row["tokens_out"],
                    "cost_usd": cost,
                })
            result["costs"] = costs

        return result
    finally:
        conn.close()


def anonymize_old_consultations(retention_days: int = 30) -> int:
    """Anonymize consultation text older than retention_days.

    Replaces query_text and response_text with '[ANONIMIZADO]' while
    preserving all numeric/stats columns for analytics.

    Returns count of anonymized rows.
    """
    # Read retention_days from settings if available
    settings = get_all_settings()
    if "retention_days" in settings:
        try:
            retention_days = int(settings["retention_days"])
        except (ValueError, TypeError):
            pass

    conn = get_connection()
    try:
        cursor = conn.execute("""
            UPDATE bot_consultations
            SET query_text = '[ANONIMIZADO]', response_text = '[ANONIMIZADO]'
            WHERE created_at < datetime('now', ? || ' days')
              AND query_text != '[ANONIMIZADO]'
        """, (f"-{retention_days}",))
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def get_bot_consultation_by_id(consultation_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM bot_consultations WHERE id = ?",
                       (consultation_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_bot_recent_consultations(limit: int = 50) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT bc.*, bu.username, bu.first_name
        FROM bot_consultations bc
        LEFT JOIN bot_users bu ON bc.telegram_id = bu.telegram_id
        ORDER BY bc.created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_bot_consultation_feedback(consultation_id: int, feedback: str) -> bool:
    conn = get_connection()
    try:
        conn.execute("UPDATE bot_consultations SET user_feedback = ? WHERE id = ?",
                     (feedback, consultation_id))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def get_all_bot_users() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT bu.*,
            (SELECT COUNT(*) FROM bot_consultations bc WHERE bc.telegram_id = bu.telegram_id) as query_count
        FROM bot_users bu
        ORDER BY bu.last_activity DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Pricing Plans
# ─────────────────────────────────────────────

def get_all_pricing_plans(active_only: bool = False) -> list[dict]:
    conn = get_connection()
    q = "SELECT * FROM pricing_plans"
    if active_only:
        q += " WHERE is_active = 1"
    q += " ORDER BY tier, period"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_pricing_plan(plan_key: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM pricing_plans WHERE plan_key = ?", (plan_key,)).fetchone()
    conn.close()
    return dict(row) if row else None


def update_pricing_plan(plan_id: int, **kwargs) -> bool:
    conn = get_connection()
    allowed = {"label", "usd_price", "mxn_price", "stripe_price_id", "paypal_plan_id",
               "mp_preapproval_id", "clip_plan_id", "is_active"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            updates.append(f"{key} = ?")
            params.append(val)
    if updates:
        updates.append("updated_at = datetime('now')")
        params.append(plan_id)
        conn.execute(f"UPDATE pricing_plans SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()
    return bool(updates)


def get_plan_prices_for_bot() -> dict:
    """Return pricing in the format bot.py expects: {plan_key: {usd, mxn, label, period}}."""
    plans = get_all_pricing_plans(active_only=True)
    result = {}
    for p in plans:
        period_label = "mes" if p["period"] == "monthly" else "año"
        result[p["plan_key"]] = {
            "usd": p["usd_price"],
            "mxn": p["mxn_price"],
            "label": p["label"],
            "period": period_label,
        }
    return result


# ─────────────────────────────────────────────
# Promotions
# ─────────────────────────────────────────────

def get_all_promotions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM promotions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_promotion(code: str, description: str = "", discount_percent: int = 0,
                     discount_amount_usd: float = 0, valid_until: str = None,
                     max_uses: int = 0, applies_to: str = "all") -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO promotions (code, description, discount_percent, discount_amount_usd,
                                    valid_until, max_uses, applies_to)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (code.upper(), description, discount_percent, discount_amount_usd,
              valid_until, max_uses, applies_to))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_promotion(promo_id: int, **kwargs) -> bool:
    conn = get_connection()
    allowed = {"code", "description", "discount_percent", "discount_amount_usd",
               "valid_until", "max_uses", "applies_to", "is_active"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            updates.append(f"{key} = ?")
            params.append(val)
    if updates:
        params.append(promo_id)
        conn.execute(f"UPDATE promotions SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()
    return bool(updates)


def delete_promotion(promo_id: int) -> bool:
    conn = get_connection()
    conn.execute("DELETE FROM promotions WHERE id = ?", (promo_id,))
    conn.commit()
    deleted = conn.total_changes > 0
    conn.close()
    return deleted


def validate_promo_code(code: str) -> dict | None:
    """Validate a promo code and return it if valid."""
    conn = get_connection()
    row = conn.execute("""
        SELECT * FROM promotions
        WHERE code = ? AND is_active = 1
        AND (max_uses = 0 OR used_count < max_uses)
        AND (valid_until IS NULL OR valid_until >= datetime('now'))
    """, (code.upper(),)).fetchone()
    conn.close()
    return dict(row) if row else None


def use_promo_code(promo_id: int):
    conn = get_connection()
    conn.execute("UPDATE promotions SET used_count = used_count + 1 WHERE id = ?", (promo_id,))
    conn.commit()
    conn.close()


def validate_and_use_promo_code(promo_id: int) -> bool:
    """Atomic: re-validate promo is still valid + increment used_count.
    Returns True if successful, False if promo expired/maxed since user applied it."""
    conn = get_connection()
    try:
        cursor = conn.execute("""
            UPDATE promotions SET used_count = used_count + 1
            WHERE id = ? AND is_active = 1
            AND (max_uses = 0 OR used_count < max_uses)
            AND (valid_until IS NULL OR valid_until >= datetime('now'))
        """, (promo_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def update_bot_user_promo(telegram_id: int, promo_id: int):
    """Store which promo code a user redeemed."""
    conn = get_connection()
    conn.execute("UPDATE bot_users SET applied_promo_id = ? WHERE telegram_id = ?",
                 (promo_id, telegram_id))
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Verification Documents
# ─────────────────────────────────────────────

def create_verification_doc(telegram_id: int, doc_type: str, file_path: str) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO verification_documents (telegram_id, doc_type, file_path)
            VALUES (?, ?, ?)
        """, (telegram_id, doc_type, file_path))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_pending_verifications() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT vd.*, bu.username, bu.first_name, bu.last_name, bu.email
        FROM verification_documents vd
        JOIN bot_users bu ON vd.telegram_id = bu.telegram_id
        WHERE vd.status = 'pending'
        ORDER BY vd.created_at ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_verification_docs(telegram_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM verification_documents
        WHERE telegram_id = ? ORDER BY created_at DESC
    """, (telegram_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def review_verification(doc_id: int, status: str, admin_notes: str = "") -> bool:
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE verification_documents
            SET status = ?, admin_notes = ?, reviewed_at = datetime('now')
            WHERE id = ?
        """, (status, admin_notes, doc_id))
        # If approved, check if user has both docs approved
        row = conn.execute("SELECT telegram_id FROM verification_documents WHERE id = ?",
                           (doc_id,)).fetchone()
        if row and status == "approved":
            tid = row["telegram_id"]
            approved = conn.execute("""
                SELECT COUNT(DISTINCT doc_type) as cnt FROM verification_documents
                WHERE telegram_id = ? AND status = 'approved'
            """, (tid,)).fetchone()["cnt"]
            if approved >= 2:  # Both cedula and INE approved
                conn.execute("""
                    UPDATE bot_users SET is_verified = 1, verification_status = 'approved',
                        verified_at = datetime('now')
                    WHERE telegram_id = ?
                """, (tid,))
        elif row and status == "rejected":
            tid = row["telegram_id"]
            conn.execute("""
                UPDATE bot_users SET verification_status = 'rejected',
                    verification_notes = ?
                WHERE telegram_id = ?
            """, (admin_notes, tid))
        conn.commit()
        return True
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Referral Rewards
# ─────────────────────────────────────────────

def process_referral(referrer_code: str, referred_telegram_id: int) -> dict | None:
    """Process a referral: link referred user and create pending reward."""
    conn = get_connection()
    try:
        # Find referrer by code
        referrer = conn.execute(
            "SELECT telegram_id FROM bot_users WHERE referral_code = ?",
            (referrer_code,)
        ).fetchone()
        if not referrer:
            return None
        referrer_id = referrer["telegram_id"]
        if referrer_id == referred_telegram_id:
            return None  # Can't refer yourself
        # Check if already referred
        existing = conn.execute(
            "SELECT referred_by FROM bot_users WHERE telegram_id = ?",
            (referred_telegram_id,)
        ).fetchone()
        if existing and existing["referred_by"]:
            return None  # Already referred by someone
        # Link referred_by
        conn.execute("UPDATE bot_users SET referred_by = ? WHERE telegram_id = ?",
                     (referrer_id, referred_telegram_id))
        # Create pending reward
        conn.execute("""
            INSERT INTO referral_rewards (referrer_id, referred_id, status, bonus_type)
            VALUES (?, ?, 'pending', 'free_queries')
        """, (referrer_id, referred_telegram_id))
        conn.commit()
        return {"referrer_id": referrer_id, "referred_id": referred_telegram_id}
    finally:
        conn.close()


def activate_referral_reward(referred_telegram_id: int) -> bool:
    """Activate reward when referred user subscribes."""
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE referral_rewards SET status = 'earned'
            WHERE referred_id = ? AND status = 'pending'
        """, (referred_telegram_id,))
        conn.commit()
        return conn.total_changes > 0
    finally:
        conn.close()


def get_referral_stats() -> dict:
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) as cnt FROM referral_rewards").fetchone()["cnt"]
        pending = conn.execute("SELECT COUNT(*) as cnt FROM referral_rewards WHERE status='pending'").fetchone()["cnt"]
        earned = conn.execute("SELECT COUNT(*) as cnt FROM referral_rewards WHERE status='earned'").fetchone()["cnt"]
        claimed = conn.execute("SELECT COUNT(*) as cnt FROM referral_rewards WHERE status='claimed'").fetchone()["cnt"]
        top_referrers = conn.execute("""
            SELECT bu.telegram_id, bu.username, bu.first_name, COUNT(*) as referral_count
            FROM referral_rewards rr
            JOIN bot_users bu ON rr.referrer_id = bu.telegram_id
            GROUP BY rr.referrer_id
            ORDER BY referral_count DESC LIMIT 10
        """).fetchall()
        return {
            "total": total, "pending": pending, "earned": earned, "claimed": claimed,
            "top_referrers": [dict(r) for r in top_referrers],
        }
    finally:
        conn.close()


def get_user_referrals(telegram_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT rr.*, bu.username, bu.first_name
        FROM referral_rewards rr
        JOIN bot_users bu ON rr.referred_id = bu.telegram_id
        WHERE rr.referrer_id = ?
        ORDER BY rr.created_at DESC
    """, (telegram_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────
# Broadcast Messages
# ─────────────────────────────────────────────

def create_broadcast(title: str, message: str, target: str = "all") -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO broadcast_messages (title, message, target)
            VALUES (?, ?, ?)
        """, (title, message, target))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_all_broadcasts(limit: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM broadcast_messages ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_broadcast_status(broadcast_id: int, status: str, sent_count: int = 0,
                             failed_count: int = 0):
    conn = get_connection()
    updates = ["status = ?"]
    params = [status]
    if sent_count:
        updates.append("sent_count = ?"); params.append(sent_count)
    if failed_count:
        updates.append("failed_count = ?"); params.append(failed_count)
    if status == "sent":
        updates.append("sent_at = datetime('now')")
    params.append(broadcast_id)
    conn.execute(f"UPDATE broadcast_messages SET {', '.join(updates)} WHERE id = ?", params)
    conn.commit()
    conn.close()


def get_broadcast_targets(target: str = "all") -> list[int]:
    """Get list of telegram_ids for a broadcast target group."""
    conn = get_connection()
    if target == "subscribers":
        rows = conn.execute(
            "SELECT telegram_id FROM bot_users WHERE subscription_status = 'active'"
        ).fetchall()
    elif target == "premium":
        rows = conn.execute(
            "SELECT telegram_id FROM bot_users WHERE subscription_plan = 'premium' AND subscription_status = 'active'"
        ).fetchall()
    elif target == "verified":
        rows = conn.execute(
            "SELECT telegram_id FROM bot_users WHERE is_verified = 1"
        ).fetchall()
    else:  # "all"
        rows = conn.execute("SELECT telegram_id FROM bot_users").fetchall()
    conn.close()
    return [r["telegram_id"] for r in rows]


# ─────────────────────────────────────────────
# Congress Events
# ─────────────────────────────────────────────

def get_all_congress_events(active_only: bool = False) -> list[dict]:
    conn = get_connection()
    q = "SELECT * FROM congress_events"
    if active_only:
        q += " WHERE is_active = 1"
    q += " ORDER BY start_date"
    rows = conn.execute(q).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_upcoming_congresses(days_ahead: int = 90) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM congress_events
        WHERE is_active = 1 AND start_date >= date('now')
        AND start_date <= date('now', '+' || ? || ' days')
        ORDER BY start_date
    """, (days_ahead,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_congress_event(name: str, short_name: str, society: str = "",
                           location: str = "", start_date: str = "", end_date: str = "",
                           description: str = "", url: str = "",
                           alert_days_before: int = 7) -> int:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO congress_events (name, short_name, society, location,
                start_date, end_date, description, url, alert_days_before)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, short_name, society, location, start_date, end_date,
              description, url, alert_days_before))
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_congress_event(event_id: int, **kwargs) -> bool:
    conn = get_connection()
    allowed = {"name", "short_name", "society", "location", "start_date", "end_date",
               "description", "url", "alert_days_before", "is_active", "alert_sent"}
    updates, params = [], []
    for key, val in kwargs.items():
        if key in allowed and val is not None:
            updates.append(f"{key} = ?"); params.append(val)
    if updates:
        params.append(event_id)
        conn.execute(f"UPDATE congress_events SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()
    conn.close()
    return bool(updates)


def delete_congress_event(event_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute("DELETE FROM congress_events WHERE id = ?", (event_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def get_congresses_needing_alert() -> list[dict]:
    """Get congresses that need alert (within alert_days_before and not yet sent)."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM congress_events
        WHERE is_active = 1 AND alert_sent = 0
        AND date(start_date, '-' || alert_days_before || ' days') <= date('now')
        AND start_date >= date('now')
        ORDER BY start_date
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
