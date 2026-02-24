"""
MedExpert Admin - Main Application
Service provider tool for managing experts, clients, guidelines, and distribution.

Usage:
    python app.py              # Start admin server
    python app.py --port 8080  # Custom port
"""

import os
import sys
import json
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import database as db
from rag_engine import get_rag_for_expert
from utils import generate_slug

console = Console()
load_dotenv(override=True)

PORT = int(os.getenv("ADMIN_PORT", 8080))

# ─────────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────────

app = FastAPI(title="MedExpert Admin", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


# ─────────────────────────────────────────────
# Dashboard
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    experts = db.get_all_experts()
    clients = db.get_all_clients()

    # Compute stats per expert
    expert_stats = []
    for e in experts:
        rag = get_rag_for_expert(e["slug"])
        guidelines = rag.list_guidelines()
        sources = db.get_web_sources_for_expert(e["id"])
        expert_stats.append({
            **e,
            "chunks": rag.get_total_count(),
            "guidelines_count": len(guidelines),
            "sources_count": len(sources),
        })

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "active_page": "dashboard",
        "experts": expert_stats,
        "clients": clients,
        "total_experts": len(experts),
        "total_clients": len(clients),
    })


# ─────────────────────────────────────────────
# Experts
# ─────────────────────────────────────────────

@app.get("/experts", response_class=HTMLResponse)
async def experts_page(request: Request):
    experts = db.get_all_experts()
    expert_data = []
    for e in experts:
        rag = get_rag_for_expert(e["slug"])
        sources = db.get_web_sources_for_expert(e["id"])
        expert_data.append({
            **e,
            "chunks": rag.get_total_count(),
            "sources_count": len(sources),
        })
    return templates.TemplateResponse("experts.html", {
        "request": request,
        "active_page": "experts",
        "experts": expert_data,
    })


@app.post("/api/experts")
async def create_expert(request: Request):
    data = await request.json()
    name = data.get("name", "").strip()
    if not name:
        return JSONResponse({"ok": False, "error": "Nombre requerido"}, status_code=400)
    slug = generate_slug(name)
    system_prompt = data.get("system_prompt", "")
    icon = data.get("icon", "&#9678;")
    try:
        expert_id = db.create_expert(name, slug, system_prompt, icon)
        # Create directories
        Path(f"data/experts/{slug}/chromadb").mkdir(parents=True, exist_ok=True)
        Path(f"data/experts/{slug}/guides").mkdir(parents=True, exist_ok=True)
        return JSONResponse({"ok": True, "id": expert_id, "slug": slug})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.put("/api/experts/{expert_id}")
async def update_expert(expert_id: int, request: Request):
    data = await request.json()
    db.update_expert(
        expert_id,
        name=data.get("name"),
        system_prompt=data.get("system_prompt"),
        icon=data.get("icon"),
    )
    return JSONResponse({"ok": True})


@app.delete("/api/experts/{expert_id}")
async def delete_expert(expert_id: int):
    deleted = db.delete_expert(expert_id)
    if not deleted:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ─────────────────────────────────────────────
# Guidelines (per expert)
# ─────────────────────────────────────────────

@app.get("/experts/{expert_id}/guidelines", response_class=HTMLResponse)
async def guidelines_page(request: Request, expert_id: int):
    expert = db.get_expert_by_id(expert_id)
    if not expert:
        return RedirectResponse("/experts")

    rag = get_rag_for_expert(expert["slug"])
    guidelines = rag.list_guidelines()
    sources = db.get_web_sources_for_expert(expert_id)

    return templates.TemplateResponse("guidelines.html", {
        "request": request,
        "active_page": "experts",
        "expert": expert,
        "guidelines": guidelines,
        "total_chunks": rag.get_total_count(),
        "sources": sources,
    })


@app.post("/api/experts/{expert_id}/guidelines/upload")
async def upload_guideline(expert_id: int, file: UploadFile = File(...)):
    expert = db.get_expert_by_id(expert_id)
    if not expert:
        return JSONResponse({"ok": False, "error": "Expert not found"}, status_code=404)

    guides_dir = Path(f"data/experts/{expert['slug']}/guides")
    guides_dir.mkdir(parents=True, exist_ok=True)

    filepath = guides_dir / file.filename
    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # Index into RAG
    rag = get_rag_for_expert(expert["slug"])
    from load_guidelines import load_file
    chunks = load_file(str(filepath), rag)

    return JSONResponse({"ok": True, "filename": file.filename, "chunks": chunks})


@app.delete("/api/experts/{expert_id}/guidelines/{source_name}")
async def delete_guideline(expert_id: int, source_name: str):
    expert = db.get_expert_by_id(expert_id)
    if not expert:
        return JSONResponse({"ok": False, "error": "Expert not found"}, status_code=404)

    rag = get_rag_for_expert(expert["slug"])
    count = rag.delete_guideline(source_name)
    return JSONResponse({"ok": True, "deleted_chunks": count})


@app.post("/api/experts/{expert_id}/guidelines/reload")
async def reload_guidelines(expert_id: int):
    expert = db.get_expert_by_id(expert_id)
    if not expert:
        return JSONResponse({"ok": False, "error": "Expert not found"}, status_code=404)

    rag = get_rag_for_expert(expert["slug"])
    rag.reload_all()
    return JSONResponse({"ok": True, "total_chunks": rag.get_total_count()})


# ─────────────────────────────────────────────
# Web Sources / Scraper
# ─────────────────────────────────────────────

@app.post("/api/web-sources")
async def create_web_source(request: Request):
    data = await request.json()
    try:
        source_id = db.create_web_source(
            expert_id=data["expert_id"],
            name=data["name"],
            url=data["url"],
            source_type=data.get("source_type", "public"),
            category=data.get("category", ""),
            css_selector_content=data.get("css_selector_content", ""),
            notes=data.get("notes", ""),
            crawl_depth=data.get("crawl_depth", 0),
            url_pattern=data.get("url_pattern", ""),
            login_url=data.get("login_url", ""),
            login_username=data.get("login_username", ""),
            login_password=data.get("login_password", ""),
            url_exclude=data.get("url_exclude", ""),
        )
        return JSONResponse({"ok": True, "id": source_id})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/web-sources/{source_id}/fetch")
async def fetch_web_source(source_id: int):
    """Trigger fetching/scraping a web source."""
    source = db.get_web_source_by_id(source_id)
    if not source:
        return JSONResponse({"ok": False, "error": "Source not found"}, status_code=404)

    expert = db.get_expert_by_id(source["expert_id"])
    if not expert:
        return JSONResponse({"ok": False, "error": "Expert not found"}, status_code=404)

    try:
        from web_scraper import WebScraper
        scraper = WebScraper()
        rag = get_rag_for_expert(expert["slug"])
        guides_dir = Path(f"data/experts/{expert['slug']}/guides")
        guides_dir.mkdir(parents=True, exist_ok=True)

        result = scraper.fetch_source(source, str(guides_dir), rag)
        db.update_web_source_status(
            source_id,
            status="active" if result.get("ok") else "error",
            error_message=result.get("error", ""),
            last_fetched=datetime.now().isoformat(),
        )
        return JSONResponse(result)
    except Exception as e:
        db.update_web_source_status(source_id, status="error", error_message=str(e))
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.delete("/api/web-sources/{source_id}")
async def delete_web_source(source_id: int):
    deleted = db.delete_web_source(source_id)
    if not deleted:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return JSONResponse({"ok": True})


# ─────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────

@app.get("/clients", response_class=HTMLResponse)
async def clients_page(request: Request):
    clients = db.get_all_clients()
    experts = db.get_all_experts()

    client_data = []
    for c in clients:
        assigned = db.get_client_experts(c["id"])
        client_data.append({
            **c,
            "assigned_experts": assigned,
            "num_experts": len(assigned),
        })

    return templates.TemplateResponse("clients.html", {
        "request": request,
        "active_page": "clients",
        "clients": client_data,
        "experts": experts,
    })


@app.post("/api/clients")
async def api_create_client(request: Request):
    data = await request.json()
    name = data.get("name", "").strip()
    hostname = data.get("hostname", "").strip()
    if not name or not hostname:
        return JSONResponse({"ok": False, "error": "Nombre y hostname requeridos"}, status_code=400)
    try:
        client_id = db.create_client(
            name=name,
            hostname=hostname,
            plan=data.get("plan", "basico"),
            tailscale_ip=data.get("tailscale_ip", ""),
            notes=data.get("notes", ""),
        )
        return JSONResponse({"ok": True, "id": client_id})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.put("/api/clients/{client_id}")
async def api_update_client(client_id: int, request: Request):
    data = await request.json()
    db.update_client(client_id, **data)
    return JSONResponse({"ok": True})


@app.delete("/api/clients/{client_id}")
async def api_delete_client(client_id: int):
    deleted = db.delete_client(client_id)
    if not deleted:
        return JSONResponse({"ok": False, "error": "Not found"}, status_code=404)
    return JSONResponse({"ok": True})


@app.post("/api/clients/{client_id}/assign-expert")
async def assign_expert(client_id: int, request: Request):
    data = await request.json()
    expert_id = data.get("expert_id")
    if not expert_id:
        return JSONResponse({"ok": False, "error": "expert_id required"}, status_code=400)
    db.assign_expert_to_client(client_id, expert_id)
    return JSONResponse({"ok": True})


@app.delete("/api/clients/{client_id}/experts/{expert_id}")
async def unassign_expert(client_id: int, expert_id: int):
    db.remove_expert_from_client(client_id, expert_id)
    return JSONResponse({"ok": True})


# ─────────────────────────────────────────────
# Distribution
# ─────────────────────────────────────────────

@app.post("/api/clients/{client_id}/push-all")
async def push_all(client_id: int):
    from distributor import push_all_to_client
    result = push_all_to_client(client_id)
    return JSONResponse({"ok": True, "results": result})


@app.post("/api/clients/{client_id}/push-chromadb/{expert_slug}")
async def push_chromadb(client_id: int, expert_slug: str):
    from distributor import push_chromadb_to_client
    result = push_chromadb_to_client(client_id, expert_slug)
    return JSONResponse(result)


@app.post("/api/clients/{client_id}/push-config")
async def push_config(client_id: int):
    from distributor import push_config_to_client
    result = push_config_to_client(client_id)
    return JSONResponse(result)


@app.post("/api/clients/{client_id}/push-license")
async def push_license(client_id: int):
    from distributor import push_license_to_client
    result = push_license_to_client(client_id)
    return JSONResponse(result)


@app.get("/api/clients/{client_id}/license")
async def get_license(client_id: int):
    from license_server import generate_license
    try:
        license_data = generate_license(client_id)
        return JSONResponse(license_data)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/clients/{client_id}/config")
async def get_config(client_id: int):
    from license_server import generate_config
    try:
        config_data = generate_config(client_id)
        return JSONResponse(config_data)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/experts/{expert_slug}/package")
async def package_expert(expert_slug: str):
    from distributor import package_expert_for_client
    try:
        path = package_expert_for_client(expert_slug)
        return JSONResponse({"ok": True, "path": str(path), "size_mb": round(path.stat().st_size / 1024 / 1024, 1)})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


# ─────────────────────────────────────────────
# License Validation (for clients to call)
# ─────────────────────────────────────────────

@app.post("/api/validate-license")
async def validate_license(request: Request):
    from license_server import validate_license_request
    data = await request.json()
    result = validate_license_request(
        license_key=data.get("license_key", ""),
        hostname=data.get("hostname", ""),
    )
    return JSONResponse(result)


# ─────────────────────────────────────────────
# Config page
# ─────────────────────────────────────────────

AVAILABLE_MODELS = {
    "anthropic": [
        {"id": "claude-opus-4-6", "name": "Opus 4.6"},
        {"id": "claude-sonnet-4-6", "name": "Sonnet 4.6"},
        {"id": "claude-sonnet-4-20250514", "name": "Sonnet 4"},
        {"id": "claude-haiku-4-5-20251001", "name": "Haiku 4.5"},
    ],
    "openai": [
        {"id": "gpt-5.1", "name": "GPT-5.1"},
        {"id": "gpt-4.1", "name": "GPT-4.1"},
        {"id": "gpt-4.1-mini", "name": "GPT-4.1 Mini"},
        {"id": "gpt-4.1-nano", "name": "GPT-4.1 Nano"},
    ],
}


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    api_keys = db.get_api_keys()
    settings = db.get_all_settings()
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    default_provider = settings.get("default_provider", "anthropic" if has_anthropic else "openai")
    default_model = settings.get("default_model", "claude-sonnet-4-20250514" if has_anthropic else "gpt-4.1")
    # Build available models based on configured keys
    models = {}
    if has_anthropic:
        models["anthropic"] = AVAILABLE_MODELS["anthropic"]
    if has_openai:
        models["openai"] = AVAILABLE_MODELS["openai"]
    # Mask keys for display (show first 7 + last 4 chars)
    def mask_key(key: str) -> str:
        if not key or len(key) < 12:
            return ""
        return key[:7] + "*" * (len(key) - 11) + key[-4:]

    anthropic_raw = os.getenv("ANTHROPIC_API_KEY", "")
    openai_raw = os.getenv("OPENAI_API_KEY", "")

    return templates.TemplateResponse("config.html", {
        "request": request,
        "active_page": "config",
        "api_keys": api_keys,
        "anthropic_key": has_anthropic,
        "openai_key": has_openai,
        "anthropic_key_masked": mask_key(anthropic_raw),
        "openai_key_masked": mask_key(openai_raw),
        "default_provider": default_provider,
        "default_model": default_model,
        "available_models": models,
    })


@app.post("/api/settings/model")
async def save_model_settings(request: Request):
    data = await request.json()
    provider = data.get("provider", "")
    model = data.get("model", "")
    if not provider or not model:
        return JSONResponse({"ok": False, "error": "Provider y modelo requeridos"}, status_code=400)
    db.set_setting("default_provider", provider)
    db.set_setting("default_model", model)
    return JSONResponse({"ok": True, "provider": provider, "model": model})


@app.post("/api/settings/api-keys")
async def save_api_keys(request: Request):
    """Save API keys to .env file and update environment."""
    data = await request.json()
    anthropic_key = data.get("anthropic_key", "").strip()
    openai_key = data.get("openai_key", "").strip()

    # Skip masked values (don't overwrite with asterisks)
    if anthropic_key and not anthropic_key.startswith("sk-"):
        anthropic_key = ""
    if openai_key and not openai_key.startswith("sk-"):
        openai_key = ""

    # Read existing .env
    env_path = Path(".env")
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()

    # Update or add keys
    def update_env(lines, key, value):
        found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}=") or line.startswith(f"# {key}="):
                if value:
                    lines[i] = f"{key}={value}"
                found = True
                break
        if not found and value:
            lines.append(f"{key}={value}")
        return lines

    if anthropic_key:
        env_lines = update_env(env_lines, "ANTHROPIC_API_KEY", anthropic_key)
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key
    if openai_key:
        env_lines = update_env(env_lines, "OPENAI_API_KEY", openai_key)
        os.environ["OPENAI_API_KEY"] = openai_key

    env_path.write_text("\n".join(env_lines) + "\n")
    return JSONResponse({"ok": True})


@app.post("/api/settings/test-api")
async def test_api_connection(request: Request):
    """Test an API key by making a minimal LLM call."""
    import time as _time
    data = await request.json()
    provider = data.get("provider", "")
    api_key = data.get("api_key", "").strip()

    # Use provided key or fall back to env
    if not api_key or not api_key.startswith("sk-"):
        api_key = os.getenv(
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", ""
        )

    if not api_key:
        return JSONResponse({"ok": False, "error": "No API key"})

    start = _time.time()
    try:
        if provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": "Responde solo 'OK'"}],
                timeout=15.0,
            )
            elapsed = _time.time() - start
            return JSONResponse({
                "ok": True,
                "model": "claude-haiku-4-5-20251001",
                "response": resp.content[0].text,
                "time": f"{elapsed:.1f}s",
            })
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4.1-nano",
                max_tokens=10,
                messages=[{"role": "user", "content": "Responde solo 'OK'"}],
                timeout=15.0,
            )
            elapsed = _time.time() - start
            return JSONResponse({
                "ok": True,
                "model": "gpt-4.1-nano",
                "response": resp.choices[0].message.content,
                "time": f"{elapsed:.1f}s",
            })
        else:
            return JSONResponse({"ok": False, "error": f"Provider desconocido: {provider}"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedExpert Admin")
    parser.add_argument("--port", type=int, default=None, help="Web server port")
    args = parser.parse_args()

    if args.port:
        PORT = args.port

    db.init_db()

    # Ensure expert directories
    for expert in db.get_all_experts():
        Path(f"data/experts/{expert['slug']}/chromadb").mkdir(parents=True, exist_ok=True)
        Path(f"data/experts/{expert['slug']}/guides").mkdir(parents=True, exist_ok=True)

    console.print(Panel(
        "[bold]MedExpert Admin[/bold]\n\n"
        f"  Web      -> http://localhost:{PORT}\n"
        f"  Experts  -> {len(db.get_all_experts())}\n"
        f"  Clients  -> {len(db.get_all_clients())}",
        border_style="blue",
    ))

    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
