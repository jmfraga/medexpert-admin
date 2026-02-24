"""
MedExpert Admin - License Server
Generates and validates licenses for client devices.
License format is a JSON file (license.json) that gets pushed to clients.
"""

import json
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from rich.console import Console

import database as db

console = Console()


def generate_license(client_id: int, duration_days: int = 365) -> dict:
    """Generate a license.json for a client device."""
    client = db.get_client_by_id(client_id)
    if not client:
        raise ValueError(f"Client {client_id} not found")

    # Get assigned experts
    client_experts = db.get_client_experts(client_id)
    expert_slugs = [e["slug"] for e in client_experts]

    expires = (datetime.now() + timedelta(days=duration_days)).isoformat()

    # Update client's license expiry in DB
    db.update_client(client_id, license_expires=expires)

    license_data = {
        "version": "1.0",
        "client_id": client["id"],
        "client_name": client["name"],
        "hostname": client["hostname"],
        "plan": client["plan"],
        "license_key": client["license_key"],
        "expires": expires,
        "issued": datetime.now().isoformat(),
        "experts_allowed": expert_slugs,
        "max_sessions_per_day": client["max_sessions_per_day"],
        "admin_server": client.get("tailscale_ip", ""),
        "offline_grace_days": 7,
        "api_keys": _get_api_keys_for_client(client),
        "default_provider": "anthropic",
    }

    return license_data


def _get_api_keys_for_client(client: dict) -> dict:
    """Get API keys to inject into client's license.
    Reads actual keys from environment or admin config."""
    import os
    keys = {}

    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if anthropic_key:
        keys["anthropic"] = anthropic_key
    if openai_key:
        keys["openai"] = openai_key

    return keys


def generate_config(client_id: int) -> dict:
    """Generate a config.json for a client device."""
    client = db.get_client_by_id(client_id)
    if not client:
        raise ValueError(f"Client {client_id} not found")

    client_experts = db.get_client_experts(client_id)

    experts_config = []
    for e in client_experts:
        experts_config.append({
            "name": e["name"],
            "slug": e["slug"],
            "icon": e.get("icon", "&#9678;"),
            "system_prompt": e.get("system_prompt", ""),
        })

    config = {
        "experts": experts_config,
        "whisper_model": "medium",
        "whisper_language": "es",
        "web_port": 8081,
        "tts_enabled": False,
        "silence_threshold": 3.0,
        "max_context_minutes": 30,
        "min_consult_interval": 15,
        "segments_trigger": 3,
    }

    return config


def save_license_file(client_id: int, output_dir: str = None) -> Path:
    """Generate and save license.json to a file."""
    license_data = generate_license(client_id)

    if output_dir is None:
        client = db.get_client_by_id(client_id)
        output_dir = f"data/clients/{client['hostname']}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / "license.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(license_data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]License saved: {filepath}[/green]")
    return filepath


def save_config_file(client_id: int, output_dir: str = None) -> Path:
    """Generate and save config.json to a file."""
    config_data = generate_config(client_id)

    if output_dir is None:
        client = db.get_client_by_id(client_id)
        output_dir = f"data/clients/{client['hostname']}"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    filepath = output_path / "config.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    console.print(f"[green]Config saved: {filepath}[/green]")
    return filepath


def validate_license_request(license_key: str, hostname: str) -> dict:
    """Validate a license request from a client device.
    Returns validation result."""
    client = db.get_client_by_hostname(hostname)
    if not client:
        return {"valid": False, "error": "Client not registered"}

    if client["license_key"] != license_key:
        return {"valid": False, "error": "Invalid license key"}

    if client["status"] != "active":
        return {"valid": False, "error": f"Client status: {client['status']}"}

    if client["license_expires"]:
        try:
            expires = datetime.fromisoformat(client["license_expires"])
            if datetime.now() > expires:
                return {"valid": False, "error": "License expired"}
        except (ValueError, TypeError):
            pass

    # Update last_seen
    db.update_client(client["id"], last_seen=datetime.now().isoformat())

    return {
        "valid": True,
        "plan": client["plan"],
        "expires": client["license_expires"],
        "client_name": client["name"],
    }
