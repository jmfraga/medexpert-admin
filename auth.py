"""
MedExpert Admin - Authentication Module
Cookie-based session auth with role-based access control.
"""

import os
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

import bcrypt as _bcrypt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse, JSONResponse

logger = logging.getLogger("medexpert.admin.auth")

DB_PATH = Path("data/medexpert_admin.db")


# ─────────────────────────────────────────────
# Password hashing
# ─────────────────────────────────────────────

def hash_password(password: str) -> str:
    return _bcrypt.hashpw(password.encode(), _bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    return _bcrypt.checkpw(password.encode(), password_hash.encode())


# ─────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────

def _get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create admin_users table and seed default admin from env vars."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS admin_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'admin' CHECK(role IN ('admin', 'soporte')),
            is_active INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            last_login TEXT DEFAULT NULL
        )
    """)
    conn.commit()

    username = os.getenv("ADMIN_USERNAME", "admin")
    password = os.getenv("ADMIN_PASSWORD")

    existing = conn.execute(
        "SELECT id FROM admin_users WHERE username = ?", (username,)
    ).fetchone()

    if not existing and password:
        conn.execute(
            "INSERT INTO admin_users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), "Administrador", "admin"),
        )
        conn.commit()
        logger.info(f"Default admin user '{username}' created")
    elif not existing and not password:
        logger.warning("ADMIN_PASSWORD not set in .env — no default admin user created")

    conn.close()


def authenticate_user(username: str, password: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM admin_users WHERE username = ? AND is_active = 1",
        (username,),
    ).fetchone()
    conn.close()
    if row and verify_password(password, row["password_hash"]):
        return dict(row)
    return None


def update_last_login(user_id: int):
    conn = _get_conn()
    conn.execute(
        "UPDATE admin_users SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(), user_id),
    )
    conn.commit()
    conn.close()


def get_all_admin_users() -> list[dict]:
    """Return all admin users (without password hashes)."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT id, username, display_name, role, is_active, created_at, last_login FROM admin_users ORDER BY id"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_admin_user(username: str, password: str, display_name: str, role: str) -> int:
    conn = _get_conn()
    cursor = conn.execute(
        "INSERT INTO admin_users (username, password_hash, display_name, role) VALUES (?, ?, ?, ?)",
        (username, hash_password(password), display_name, role),
    )
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id


def update_admin_user(user_id: int, **kwargs):
    allowed = {"display_name", "role", "is_active", "password"}
    fields, values = [], []
    for k, v in kwargs.items():
        if k not in allowed:
            continue
        if k == "password":
            fields.append("password_hash = ?")
            values.append(hash_password(v))
        else:
            fields.append(f"{k} = ?")
            values.append(v)
    if not fields:
        return
    values.append(user_id)
    conn = _get_conn()
    conn.execute(f"UPDATE admin_users SET {', '.join(fields)} WHERE id = ?", values)
    conn.commit()
    conn.close()


def delete_admin_user(user_id: int):
    """Soft delete: deactivate the user."""
    conn = _get_conn()
    conn.execute("UPDATE admin_users SET is_active = 0 WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()


def get_admin_user_by_id(user_id: int) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM admin_users WHERE id = ? AND is_active = 1", (user_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ─────────────────────────────────────────────
# Access control
# ─────────────────────────────────────────────

# No auth required
PUBLIC_PATHS = {"/login", "/favicon.ico"}
PUBLIC_PREFIXES = (
    "/static/",
    "/api/stripe/webhook",
    "/api/paypal/webhook",
    "/api/mercadopago/webhook",
    "/api/clip/webhook",
    "/api/validate-license",
)

# Soporte role — allowed pages and API prefixes
SOPORTE_ALLOWED_PAGES = {"/", "/bot", "/analytics", "/tickets"}
SOPORTE_ALLOWED_API_PREFIXES = ("/api/analytics/", "/api/tickets/")


def _is_public(path: str, method: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    for prefix in PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return True
    # Bot creates tickets via POST /api/tickets
    if path == "/api/tickets" and method == "POST":
        return True
    return False


def _soporte_allowed(path: str, method: str) -> bool:
    if path in SOPORTE_ALLOWED_PAGES:
        return True
    for prefix in SOPORTE_ALLOWED_API_PREFIXES:
        if path.startswith(prefix):
            return True
    # Soporte can read bot data (GET only)
    if path.startswith("/api/bot/") and method == "GET":
        return True
    return False


# ─────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.url.path
        method = request.method

        # Public paths — no auth
        if _is_public(path, method):
            return await call_next(request)

        # Check session
        user_id = request.session.get("user_id")
        if not user_id:
            if path.startswith("/api/"):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return RedirectResponse("/login", status_code=302)

        # Load user
        user = get_admin_user_by_id(user_id)
        if not user:
            request.session.clear()
            return RedirectResponse("/login", status_code=302)

        # Role-based access for soporte
        if user["role"] == "soporte" and not _soporte_allowed(path, method):
            if path.startswith("/api/"):
                return JSONResponse({"error": "Forbidden"}, status_code=403)
            return RedirectResponse("/", status_code=302)

        # Inject user into request scope (picked up by template patch)
        request.scope["user"] = user

        return await call_next(request)


def get_current_user(request) -> dict | None:
    """Helper to get user from request scope."""
    return request.scope.get("user")
