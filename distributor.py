"""
MedExpert Admin - Distributor
Handles distribution of ChromaDB, configs, and licenses to client devices.
Uses rsync/scp over Tailscale for file transfer.
Includes Knowledge Pack versioning with manifest and integrity verification.
"""

import json
import hashlib
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from rich.console import Console

import database as db
from license_server import generate_license, generate_config

console = Console()

KB_VERSIONS_DIR = Path("data/kb_versions")


# ─────────────────────────────────────────────
# Knowledge Pack Manifest
# ─────────────────────────────────────────────

def _hash_file(filepath: Path) -> str:
    """SHA256 hash of a single file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _hash_directory(dirpath: Path) -> tuple[str, list[dict]]:
    """Compute SHA256 of all files in a directory. Returns (combined_hash, file_list)."""
    files = sorted(f for f in dirpath.rglob("*") if f.is_file())
    combined = hashlib.sha256()
    file_list = []
    for f in files:
        fhash = _hash_file(f)
        combined.update(fhash.encode())
        file_list.append({
            "path": str(f.relative_to(dirpath)),
            "size": f.stat().st_size,
            "sha256": fhash,
        })
    return combined.hexdigest(), file_list


def generate_manifest(expert_slug: str, version: str = None) -> dict:
    """Generate a Knowledge Pack manifest for an expert's ChromaDB.

    Returns manifest dict with:
    - kb_id, kb_version (semver), created_at
    - source_manifest: list of indexed sources with hashes
    - chromadb_hash: integrity hash of the entire chromadb directory
    - chunking_params, embedding_model, total_chunks
    """
    from rag_engine import get_rag_for_expert

    chromadb_dir = Path(f"data/experts/{expert_slug}/chromadb")
    if not chromadb_dir.exists():
        raise FileNotFoundError(f"No ChromaDB for {expert_slug}")

    rag = get_rag_for_expert(expert_slug)
    guidelines = rag.list_guidelines()
    total_chunks = rag.get_total_count()

    # Auto-increment version if not provided
    if not version:
        version = _next_version(expert_slug)

    # Hash the chromadb directory for integrity verification
    dir_hash, file_list = _hash_directory(chromadb_dir)

    # Build source manifest
    source_manifest = []
    for g in guidelines:
        source_manifest.append({
            "source": g["source"],
            "category": g.get("category", ""),
            "chunks": g["chunks"],
        })

    manifest = {
        "kb_id": f"{expert_slug}",
        "kb_version": version,
        "created_at": datetime.now().isoformat(),
        "expert_slug": expert_slug,
        "total_chunks": total_chunks,
        "total_sources": len(guidelines),
        "chromadb_hash": dir_hash,
        "chromadb_files": len(file_list),
        "chunking_params": {
            "chunk_size": 500,
            "overlap": 100,
            "method": "clinical_sections",
        },
        "embedding_model": "all-MiniLM-L6-v2",
        "source_manifest": source_manifest,
    }

    return manifest


def _next_version(expert_slug: str) -> str:
    """Auto-generate next semver version based on existing versions."""
    versions_dir = KB_VERSIONS_DIR / expert_slug
    if not versions_dir.exists():
        return "1.0.0"

    existing = []
    for f in versions_dir.glob("*/manifest.json"):
        try:
            m = json.loads(f.read_text())
            existing.append(m.get("kb_version", "0.0.0"))
        except Exception:
            pass

    if not existing:
        return "1.0.0"

    # Parse and increment patch version
    latest = sorted(existing, key=lambda v: [int(x) for x in v.split(".")])[-1]
    parts = [int(x) for x in latest.split(".")]
    parts[2] += 1  # bump patch
    return ".".join(str(p) for p in parts)


def save_kb_version(expert_slug: str, version: str = None) -> dict:
    """Create a versioned Knowledge Pack snapshot.

    Copies ChromaDB + generates manifest, keeping up to 3 versions.
    Returns the manifest.
    """
    manifest = generate_manifest(expert_slug, version)
    version = manifest["kb_version"]

    # Create version directory
    version_dir = KB_VERSIONS_DIR / expert_slug / version
    version_dir.mkdir(parents=True, exist_ok=True)

    # Copy ChromaDB to version dir
    source_chromadb = Path(f"data/experts/{expert_slug}/chromadb")
    dest_chromadb = version_dir / "chromadb"
    if dest_chromadb.exists():
        shutil.rmtree(dest_chromadb)
    shutil.copytree(source_chromadb, dest_chromadb)

    # Save manifest
    manifest_path = version_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    # Prune old versions (keep latest 3)
    _prune_old_versions(expert_slug, keep=3)

    console.print(f"[green]KB version {version} saved for {expert_slug} "
                  f"({manifest['total_chunks']} chunks, {manifest['total_sources']} sources)[/green]")

    return manifest


def _prune_old_versions(expert_slug: str, keep: int = 3):
    """Remove old KB versions, keeping the latest N."""
    versions_dir = KB_VERSIONS_DIR / expert_slug
    if not versions_dir.exists():
        return

    versions = []
    for d in versions_dir.iterdir():
        if d.is_dir() and (d / "manifest.json").exists():
            try:
                m = json.loads((d / "manifest.json").read_text())
                versions.append((m.get("created_at", ""), d))
            except Exception:
                pass

    versions.sort(key=lambda x: x[0], reverse=True)

    for _, vdir in versions[keep:]:
        console.print(f"[yellow]Pruning old KB version: {vdir.name}[/yellow]")
        shutil.rmtree(vdir)


def list_kb_versions(expert_slug: str) -> list[dict]:
    """List all available KB versions for an expert."""
    versions_dir = KB_VERSIONS_DIR / expert_slug
    if not versions_dir.exists():
        return []

    versions = []
    for d in sorted(versions_dir.iterdir()):
        manifest_path = d / "manifest.json"
        if d.is_dir() and manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text())
                m["_dir"] = str(d)
                versions.append(m)
            except Exception:
                pass

    return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)


def verify_kb_integrity(expert_slug: str, version: str) -> dict:
    """Verify integrity of a KB version by recalculating hashes.
    Returns {ok, errors}.
    """
    version_dir = KB_VERSIONS_DIR / expert_slug / version
    manifest_path = version_dir / "manifest.json"

    if not manifest_path.exists():
        return {"ok": False, "errors": ["Manifest not found"]}

    manifest = json.loads(manifest_path.read_text())
    chromadb_dir = version_dir / "chromadb"

    if not chromadb_dir.exists():
        return {"ok": False, "errors": ["ChromaDB directory not found"]}

    actual_hash, _ = _hash_directory(chromadb_dir)
    expected_hash = manifest.get("chromadb_hash", "")

    if actual_hash != expected_hash:
        return {"ok": False, "errors": [
            f"Hash mismatch: expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
        ]}

    return {"ok": True, "errors": []}


def push_chromadb_to_client(client_id: int, expert_slug: str) -> dict:
    """Push an expert's ChromaDB to a client device via rsync.
    Saves a KB version snapshot before pushing for rollback support.
    Also pushes the manifest.json for client-side integrity verification.
    """
    client = db.get_client_by_id(client_id)
    if not client:
        return {"ok": False, "error": "Client not found"}

    if not client.get("tailscale_ip"):
        return {"ok": False, "error": "No Tailscale IP configured"}

    source_dir = Path(f"data/experts/{expert_slug}/chromadb/")
    if not source_dir.exists():
        return {"ok": False, "error": f"No ChromaDB for {expert_slug}"}

    # Save KB version snapshot before pushing
    try:
        manifest = save_kb_version(expert_slug)
        kb_version = manifest["kb_version"]
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save KB version: {e}[/yellow]")
        kb_version = datetime.now().isoformat()
        manifest = None

    remote_path = f"data/experts/{expert_slug}/chromadb/"
    target = f"{client['tailscale_ip']}:~/medexpert-client/{remote_path}"

    log_id = db.log_distribution(client_id, expert_slug, "chromadb_push")

    try:
        result = subprocess.run(
            ["rsync", "-avz", "--delete", str(source_dir) + "/", target],
            capture_output=True, text=True, timeout=300,
        )

        if result.returncode == 0:
            # Push manifest.json alongside chromadb
            if manifest:
                manifest_local = source_dir.parent / "manifest.json"
                manifest_local.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False))
                manifest_target = (f"{client['tailscale_ip']}:~/medexpert-client/"
                                   f"data/experts/{expert_slug}/manifest.json")
                subprocess.run(
                    ["scp", str(manifest_local), manifest_target],
                    capture_output=True, text=True, timeout=30,
                )

            # Update sync status with KB version
            expert = db.get_expert_by_slug(expert_slug)
            if expert:
                db.update_client_expert_sync(
                    client_id, expert["id"],
                    chromadb_version=kb_version,
                )
            return {"ok": True, "output": result.stdout, "kb_version": kb_version}
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


def push_glossary_to_client(client_id: int, expert_slug: str) -> dict:
    """Push an expert's glossary terms as glossary.json to a client device."""
    client = db.get_client_by_id(client_id)
    if not client:
        return {"ok": False, "error": "Client not found"}

    if not client.get("tailscale_ip"):
        return {"ok": False, "error": "No Tailscale IP configured"}

    expert = db.get_expert_by_slug(expert_slug)
    if not expert:
        return {"ok": False, "error": f"Expert {expert_slug} not found"}

    terms = db.get_glossary_terms_for_expert(expert["id"])
    glossary_data = {
        "specialty": expert_slug,
        "version": datetime.now().strftime("%Y%m%d"),
        "terms": [t["term"] for t in terms],
    }

    # Save locally
    local_dir = Path(f"data/clients/{client['hostname']}")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / "glossary.json"
    with open(local_file, "w", encoding="utf-8") as f:
        json.dump(glossary_data, f, ensure_ascii=False, indent=2)

    target = f"{client['tailscale_ip']}:~/medexpert-client/data/glossary.json"

    try:
        result = subprocess.run(
            ["scp", str(local_file), target],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode == 0:
            db.log_distribution(client_id, expert_slug, "glossary_push", "success")
            console.print(f"[green]Glossary pushed to {client['hostname']}: {len(terms)} terms[/green]")
            return {"ok": True, "terms": len(terms)}
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
        results["experts"][f"{expert['slug']}_glossary"] = push_glossary_to_client(client_id, expert["slug"])

    return results


def package_expert_for_client(expert_slug: str, output_dir: str = None) -> Path:
    """Package an expert's ChromaDB + manifest into a tar.gz for manual distribution.
    Saves a KB version and includes manifest.json in the package.
    """
    source_dir = Path(f"data/experts/{expert_slug}/chromadb")
    if not source_dir.exists():
        raise FileNotFoundError(f"No ChromaDB for {expert_slug}")

    # Generate manifest and save alongside chromadb
    try:
        manifest = save_kb_version(expert_slug)
        manifest_path = source_dir.parent / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    except Exception as e:
        console.print(f"[yellow]Warning: Could not generate manifest: {e}[/yellow]")

    if output_dir is None:
        output_dir = "data/packages"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version = manifest.get("kb_version", "unknown") if manifest else "unknown"
    archive_name = f"{expert_slug}_kb_v{version}_{timestamp}"
    archive_path = output_path / archive_name

    # Create archive including chromadb/ and manifest.json
    expert_dir = source_dir.parent
    shutil.make_archive(str(archive_path), "gztar", str(expert_dir.parent), expert_dir.name)

    final_path = Path(f"{archive_path}.tar.gz")
    console.print(f"[green]KB package created: {final_path} "
                  f"(v{version}, {final_path.stat().st_size / 1024 / 1024:.1f} MB)[/green]")
    return final_path
