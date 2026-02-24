"""
MedExpert Admin - Distributor
Handles distribution of ChromaDB, configs, and licenses to client devices.
Uses rsync/scp over Tailscale for file transfer.
"""

import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from rich.console import Console

import database as db
from license_server import generate_license, generate_config

console = Console()


def push_chromadb_to_client(client_id: int, expert_slug: str) -> dict:
    """Push an expert's ChromaDB to a client device via rsync."""
    client = db.get_client_by_id(client_id)
    if not client:
        return {"ok": False, "error": "Client not found"}

    if not client.get("tailscale_ip"):
        return {"ok": False, "error": "No Tailscale IP configured"}

    source_dir = Path(f"data/experts/{expert_slug}/chromadb/")
    if not source_dir.exists():
        return {"ok": False, "error": f"No ChromaDB for {expert_slug}"}

    remote_path = f"data/experts/{expert_slug}/chromadb/"
    target = f"{client['tailscale_ip']}:~/medexpert-client/{remote_path}"

    log_id = db.log_distribution(client_id, expert_slug, "chromadb_push")

    try:
        result = subprocess.run(
            ["rsync", "-avz", "--delete", str(source_dir) + "/", target],
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode == 0:
            # Update sync status
            expert = db.get_expert_by_slug(expert_slug)
            if expert:
                db.update_client_expert_sync(
                    client_id, expert["id"],
                    chromadb_version=datetime.now().isoformat(),
                )
            return {"ok": True, "output": result.stdout}
        else:
            return {"ok": False, "error": result.stderr}

    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Transfer timed out (5 min)"}
    except FileNotFoundError:
        return {"ok": False, "error": "rsync not found. Install with: brew install rsync"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def push_config_to_client(client_id: int) -> dict:
    """Push config.json to a client device."""
    client = db.get_client_by_id(client_id)
    if not client:
        return {"ok": False, "error": "Client not found"}

    if not client.get("tailscale_ip"):
        return {"ok": False, "error": "No Tailscale IP configured"}

    config_data = generate_config(client_id)

    # Save locally first
    local_dir = Path(f"data/clients/{client['hostname']}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / "config.json"
    import json
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    target = f"{client['tailscale_ip']}:~/medexpert-client/data/config.json"

    try:
        result = subprocess.run(
            ["scp", str(local_file), target],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0:
            db.log_distribution(client_id, "", "config_push", "success")
            return {"ok": True}
        else:
            return {"ok": False, "error": result.stderr}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def push_license_to_client(client_id: int) -> dict:
    """Push license.json to a client device."""
    client = db.get_client_by_id(client_id)
    if not client:
        return {"ok": False, "error": "Client not found"}

    if not client.get("tailscale_ip"):
        return {"ok": False, "error": "No Tailscale IP configured"}

    license_data = generate_license(client_id)

    local_dir = Path(f"data/clients/{client['hostname']}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / "license.json"
    import json
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(license_data, f, ensure_ascii=False, indent=2)

    target = f"{client['tailscale_ip']}:~/medexpert-client/data/license.json"

    try:
        result = subprocess.run(
            ["scp", str(local_file), target],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0:
            db.log_distribution(client_id, "", "license_push", "success")
            return {"ok": True}
        else:
            return {"ok": False, "error": result.stderr}

    except Exception as e:
        return {"ok": False, "error": str(e)}


def push_all_to_client(client_id: int) -> dict:
    """Push everything (license, config, ChromaDB for all assigned experts) to a client."""
    results = {"license": None, "config": None, "experts": {}}

    results["license"] = push_license_to_client(client_id)
    results["config"] = push_config_to_client(client_id)

    client_experts = db.get_client_experts(client_id)
    for expert in client_experts:
        results["experts"][expert["slug"]] = push_chromadb_to_client(client_id, expert["slug"])

    return results


def package_expert_for_client(expert_slug: str, output_dir: str = None) -> Path:
    """Package an expert's ChromaDB into a tar.gz for manual distribution."""
    source_dir = Path(f"data/experts/{expert_slug}/chromadb")
    if not source_dir.exists():
        raise FileNotFoundError(f"No ChromaDB for {expert_slug}")

    if output_dir is None:
        output_dir = "data/packages"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{expert_slug}_chromadb_{timestamp}"
    archive_path = output_path / archive_name

    shutil.make_archive(str(archive_path), "gztar", str(source_dir.parent), "chromadb")

    final_path = Path(f"{archive_path}.tar.gz")
    console.print(f"[green]Package created: {final_path} ({final_path.stat().st_size / 1024 / 1024:.1f} MB)[/green]")
    return final_path
