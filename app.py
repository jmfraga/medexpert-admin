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
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("medexpert.admin")

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
            use_browser=1 if data.get("use_browser") else 0,
            allowed_domains=data.get("allowed_domains", ""),
            min_content_length=int(data.get("min_content_length", 2000)),
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
        from web_scraper import GuidelineScraper
        scraper = GuidelineScraper()
        rag = get_rag_for_expert(expert["slug"])
        guides_dir = Path(f"data/experts/{expert['slug']}/guides")
        guides_dir.mkdir(parents=True, exist_ok=True)

        result = await scraper.fetch_and_index(source, rag, expert["slug"])
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
# Glossary
# ─────────────────────────────────────────────

@app.get("/experts/{expert_id}/glossary", response_class=HTMLResponse)
async def glossary_page(request: Request, expert_id: int):
    expert = db.get_expert_by_id(expert_id)
    if not expert:
        return RedirectResponse("/experts")
    terms = db.get_glossary_terms_for_expert(expert_id)
    categories = {}
    for t in terms:
        cat = t["category"] or "General"
        categories.setdefault(cat, []).append(t)
    return templates.TemplateResponse("glossary.html", {
        "request": request,
        "expert": expert,
        "terms": terms,
        "categories": categories,
        "total_terms": len(terms),
        "active_page": "experts",
    })


@app.post("/api/glossary")
async def create_glossary_term(request: Request):
    data = await request.json()
    expert_id = data.get("expert_id")
    term = data.get("term", "").strip()
    category = data.get("category", "").strip()
    synonyms = data.get("synonyms", "").strip()
    if not expert_id or not term:
        return JSONResponse({"ok": False, "error": "expert_id and term required"})
    term_id = db.create_glossary_term(expert_id, term, category, synonyms)
    return JSONResponse({"ok": True, "id": term_id})


@app.put("/api/glossary/{term_id}")
async def update_glossary_term_route(term_id: int, request: Request):
    data = await request.json()
    db.update_glossary_term(term_id, term=data.get("term"), category=data.get("category"), synonyms=data.get("synonyms"))
    return JSONResponse({"ok": True})


@app.delete("/api/glossary/{term_id}")
async def delete_glossary_term_route(term_id: int):
    deleted = db.delete_glossary_term(term_id)
    return JSONResponse({"ok": deleted})


@app.post("/api/experts/{expert_id}/glossary/import")
async def import_glossary(expert_id: int, request: Request):
    data = await request.json()
    terms = data.get("terms", [])
    if not terms:
        return JSONResponse({"ok": False, "error": "No terms provided"})
    count = db.bulk_create_glossary_terms(expert_id, terms)
    return JSONResponse({"ok": True, "imported": count})


@app.get("/api/experts/{expert_id}/glossary/export")
async def export_glossary(expert_id: int):
    terms = db.get_glossary_terms_for_expert(expert_id)
    expert = db.get_expert_by_id(expert_id)
    terms_by_category = {}
    for t in terms:
        cat = t["category"] or "General"
        terms_by_category.setdefault(cat, []).append(t["term"])
    return JSONResponse({
        "specialty": expert["slug"] if expert else "",
        "version": datetime.now().strftime("%Y%m%d"),
        "terms": [t["term"] for t in terms],
        "terms_by_category": terms_by_category,
    })


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
# Tickets (from clients)
# ─────────────────────────────────────────────

@app.post("/api/tickets")
async def receive_ticket(request: Request):
    data = await request.json()
    ticket_type = data.get("ticket_type", "").strip()
    title = data.get("title", "").strip()
    if not ticket_type or not title:
        return JSONResponse({"ok": False, "error": "ticket_type and title required"})
    if ticket_type not in ("transcription", "bug", "feature"):
        return JSONResponse({"ok": False, "error": "Invalid ticket_type"})

    # Try to find client by hostname
    client_id = None
    hostname = data.get("hostname", "")
    if hostname:
        client = db.get_client_by_hostname(hostname)
        if client:
            client_id = client["id"]

    ticket_id = db.create_ticket(
        client_id=client_id,
        ticket_type=ticket_type,
        title=title,
        description=data.get("description", ""),
        expert_slug=data.get("expert_slug", ""),
        original_text=data.get("original_text", ""),
        suggested_text=data.get("suggested_text", ""),
    )
    return JSONResponse({"ok": True, "id": ticket_id})


@app.get("/tickets", response_class=HTMLResponse)
async def tickets_page(request: Request):
    status_filter = request.query_params.get("status")
    type_filter = request.query_params.get("type")
    tickets = db.get_all_tickets(status=status_filter, ticket_type=type_filter)
    stats = db.get_ticket_stats()
    # Enrich with client name or Telegram username
    for t in tickets:
        if t.get("telegram_id"):
            bot_user = db.get_bot_user(t["telegram_id"])
            t["telegram_username"] = (bot_user.get("username") or bot_user.get("first_name") or str(t["telegram_id"])) if bot_user else str(t["telegram_id"])
        if t.get("client_id"):
            client = db.get_client_by_id(t["client_id"])
            t["client_name"] = client["name"] if client else "Desconocido"
        else:
            t["client_name"] = "Desconocido"
    return templates.TemplateResponse("tickets.html", {
        "request": request,
        "tickets": tickets,
        "stats": stats,
        "active_page": "tickets",
        "current_status": status_filter or "",
        "current_type": type_filter or "",
    })


@app.put("/api/tickets/{ticket_id}")
async def update_ticket_route(ticket_id: int, request: Request):
    data = await request.json()
    updates = {}
    if "status" in data:
        updates["status"] = data["status"]
        if data["status"] == "resolved":
            updates["resolved_at"] = datetime.now().isoformat()
    if "admin_notes" in data:
        updates["admin_notes"] = data["admin_notes"]
    db.update_ticket(ticket_id, **updates)
    return JSONResponse({"ok": True})


@app.post("/api/tickets/{ticket_id}/respond")
async def respond_to_ticket(ticket_id: int, request: Request):
    """Respond to a support ticket and notify user via Telegram."""
    data = await request.json()
    response_text = data.get("response", "").strip()
    if not response_text:
        return JSONResponse({"ok": False, "error": "Response text required"})

    ticket = db.get_ticket_by_id(ticket_id)
    if not ticket:
        return JSONResponse({"ok": False, "error": "Ticket not found"})

    # Save response and mark as resolved
    db.update_ticket(ticket_id, admin_response=response_text, status="resolved",
                     resolved_at=datetime.now().isoformat())

    # Notify user via Telegram if we have their telegram_id
    telegram_id = ticket.get("telegram_id")
    if telegram_id:
        try:
            import httpx
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            if bot_token:
                msg = (
                    f"<b>Respuesta a tu ticket #{ticket_id}</b>\n\n"
                    f"{response_text}\n\n"
                    "<i>— Equipo MedExpert</i>"
                )
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        json={"chat_id": telegram_id, "text": msg, "parse_mode": "HTML"},
                    )
                logger.info(f"Ticket #{ticket_id} response sent to Telegram user {telegram_id}")
        except Exception as e:
            logger.error(f"Failed to notify Telegram user {telegram_id}: {e}")
            return JSONResponse({"ok": True, "warning": "Response saved but Telegram notification failed"})

    return JSONResponse({"ok": True})


@app.post("/api/tickets/{ticket_id}/apply-to-glossary")
async def apply_ticket_to_glossary(ticket_id: int):
    ticket = db.get_ticket_by_id(ticket_id)
    if not ticket:
        return JSONResponse({"ok": False, "error": "Ticket not found"})
    if ticket["ticket_type"] != "transcription":
        return JSONResponse({"ok": False, "error": "Not a transcription ticket"})
    if not ticket["suggested_text"]:
        return JSONResponse({"ok": False, "error": "No suggested text"})

    # Find expert
    expert = db.get_expert_by_slug(ticket["expert_slug"]) if ticket["expert_slug"] else None
    if not expert:
        # Try first expert as fallback
        experts = db.get_all_experts()
        expert = experts[0] if experts else None
    if not expert:
        return JSONResponse({"ok": False, "error": "No expert found"})

    # Add suggested text as glossary term
    term_id = db.create_glossary_term(expert["id"], ticket["suggested_text"].strip())

    # Mark ticket as resolved
    db.update_ticket(ticket_id, status="resolved", resolved_at=datetime.now().isoformat(),
                     admin_notes=(ticket.get("admin_notes", "") + "\nTermino agregado al glosario.").strip())

    return JSONResponse({"ok": True, "term_id": term_id})


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


@app.post("/api/clients/{client_id}/client-config")
async def save_client_config(client_id: int, request: Request):
    data = await request.json()
    allowed_keys = {"whisper_model", "whisper_language", "silence_threshold",
                    "min_consult_interval", "segments_trigger", "max_context_minutes"}
    config = {k: v for k, v in data.items() if k in allowed_keys}
    # Type coercion
    for k in ("silence_threshold",):
        if k in config:
            config[k] = float(config[k])
    for k in ("min_consult_interval", "segments_trigger", "max_context_minutes"):
        if k in config:
            config[k] = int(config[k])
    db.update_client(client_id, client_config=json.dumps(config))
    return JSONResponse({"ok": True, "config": config})


@app.get("/api/clients/{client_id}/client-config")
async def get_client_config(client_id: int):
    client = db.get_client_by_id(client_id)
    if not client:
        return JSONResponse({"ok": False, "error": "No encontrado"}, status_code=404)
    try:
        config = json.loads(client.get("client_config", "{}") or "{}")
    except (json.JSONDecodeError, TypeError):
        config = {}
    defaults = {
        "whisper_model": "small",
        "whisper_language": "es",
        "silence_threshold": 3.0,
        "min_consult_interval": 15,
        "segments_trigger": 3,
        "max_context_minutes": 30,
    }
    return JSONResponse({"ok": True, "config": {**defaults, **config}})


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


# ─────────────────────────────────────────────
# Telegram Bot Dashboard
# ─────────────────────────────────────────────

@app.get("/bot", response_class=HTMLResponse)
async def bot_dashboard(request: Request):
    stats = db.get_bot_stats()
    users = db.get_all_bot_users()
    consultations = db.get_bot_recent_consultations(limit=50)
    return templates.TemplateResponse("bot.html", {
        "request": request,
        "active_page": "bot",
        "stats": stats,
        "users": users,
        "consultations": consultations,
    })


@app.put("/api/bot/users/{telegram_id}/plan")
async def update_bot_user_plan(telegram_id: int, request: Request):
    data = await request.json()
    plan = data.get("plan", "free")
    notify = data.get("notify", False)

    if plan == "free":
        db.cancel_bot_user_subscription(telegram_id)
    else:
        db.update_bot_user_subscription(telegram_id, plan, "active")

    # Notify user via Telegram
    if notify:
        try:
            import httpx
            bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
            if bot_token:
                if plan == "free":
                    msg = "Tu suscripcion ha sido cancelada por el administrador.\n\nPuedes volver a suscribirte con /suscribir"
                else:
                    msg = f"Tu plan ha sido actualizado a: <b>{plan.capitalize()}</b>\n\nGracias por ser parte de MedExpert."
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        json={"chat_id": telegram_id, "text": msg, "parse_mode": "HTML"},
                    )
        except Exception as e:
            logger.error(f"Failed to notify user {telegram_id}: {e}")

    return JSONResponse({"ok": True, "plan": plan})


@app.post("/api/bot/users/{telegram_id}/notify")
async def notify_bot_user(telegram_id: int, request: Request):
    """Send a custom message to a Telegram user."""
    data = await request.json()
    message = data.get("message", "").strip()
    if not message:
        return JSONResponse({"ok": False, "error": "Message required"})

    try:
        import httpx
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            return JSONResponse({"ok": False, "error": "Bot token not configured"})

        msg = f"{message}\n\n<i>— Equipo MedExpert</i>"
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": telegram_id, "text": msg, "parse_mode": "HTML"},
            )
        if resp.status_code == 200:
            return JSONResponse({"ok": True})
        else:
            return JSONResponse({"ok": False, "error": f"Telegram API error: {resp.status_code}"})
    except Exception as e:
        logger.error(f"Failed to notify user {telegram_id}: {e}")
        return JSONResponse({"ok": False, "error": str(e)})


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
    if not api_key or "*" in api_key:
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
# Stripe Webhook
# ─────────────────────────────────────────────

@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    """Handle Stripe webhook events for subscription management."""
    import stripe

    stripe_key = os.getenv("STRIPE_SECRET_KEY")
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET")

    if not stripe_key:
        return JSONResponse({"error": "Stripe not configured"}, status_code=500)

    stripe.api_key = stripe_key
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    try:
        if webhook_secret:
            event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
        else:
            # Dev mode: parse without signature verification
            import json
            event = stripe.Event.construct_from(json.loads(payload), stripe.api_key)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        logger.error(f"Stripe webhook error: {e}")
        return JSONResponse({"error": "Invalid signature"}, status_code=400)

    # Handle checkout completed
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        telegram_id = int(session.get("client_reference_id") or
                          session.get("metadata", {}).get("telegram_id", 0))
        plan = session.get("metadata", {}).get("plan", "basic")
        customer_id = session.get("customer")

        if telegram_id:
            db.update_bot_user_subscription(
                telegram_id=telegram_id,
                plan=plan,
                status="active",
                stripe_customer_id=customer_id,
            )
            logger.info(f"Subscription activated: {telegram_id} -> {plan}")

    # Handle subscription cancelled/expired
    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub = event["data"]["object"]
        customer_id = sub.get("customer")
        status = sub.get("status")

        if status in ("canceled", "unpaid", "past_due"):
            # Find user by stripe customer ID and cancel
            conn = db.get_connection()
            try:
                row = conn.execute(
                    "SELECT telegram_id FROM bot_users WHERE stripe_customer_id = ?",
                    (customer_id,)
                ).fetchone()
                if row:
                    db.cancel_bot_user_subscription(row["telegram_id"])
                    logger.info(f"Subscription cancelled: {row['telegram_id']}")
            finally:
                conn.close()

    return JSONResponse({"received": True})


# ─────────────────────────────────────────────
# PayPal Webhook
# ─────────────────────────────────────────────

@app.post("/api/paypal/webhook")
async def paypal_webhook(request: Request):
    """Handle PayPal webhook events for subscription management."""
    data = await request.json()
    event_type = data.get("event_type", "")

    if event_type == "BILLING.SUBSCRIPTION.ACTIVATED":
        resource = data.get("resource", {})
        custom_id = resource.get("custom_id", "")

        if "_" in custom_id:
            parts = custom_id.split("_")
            telegram_id = int(parts[0])
            plan = parts[1]
            sub_id = resource.get("id", "")

            db.update_bot_user_subscription(
                telegram_id=telegram_id,
                plan=plan,
                status="active",
                stripe_customer_id=f"paypal_{sub_id}",
            )
            logger.info(f"PayPal subscription activated: {telegram_id} -> {plan}")

    elif event_type in ("BILLING.SUBSCRIPTION.CANCELLED", "BILLING.SUBSCRIPTION.SUSPENDED"):
        resource = data.get("resource", {})
        custom_id = resource.get("custom_id", "")
        if "_" in custom_id:
            telegram_id = int(custom_id.split("_")[0])
            db.cancel_bot_user_subscription(telegram_id)
            logger.info(f"PayPal subscription cancelled: {telegram_id}")

    return JSONResponse({"received": True})


# ─────────────────────────────────────────────
# Mercado Pago Webhook
# ─────────────────────────────────────────────

@app.post("/api/mercadopago/webhook")
async def mercadopago_webhook(request: Request):
    """Handle Mercado Pago IPN notifications."""
    import mercadopago

    mp_token = os.getenv("MP_ACCESS_TOKEN")
    if not mp_token:
        return JSONResponse({"error": "MP not configured"}, status_code=500)

    data = await request.json()
    action = data.get("action")
    data_id = data.get("data", {}).get("id")

    if action == "payment.created" and data_id:
        sdk = mercadopago.SDK(mp_token)
        payment = sdk.payment().get(data_id)
        payment_info = payment.get("response", {})

        status = payment_info.get("status")
        external_ref = payment_info.get("external_reference", "")

        if status == "approved" and "_" in external_ref:
            parts = external_ref.split("_")
            telegram_id = int(parts[0])
            plan = parts[1]

            db.update_bot_user_subscription(
                telegram_id=telegram_id,
                plan=plan,
                status="active",
                stripe_customer_id=f"mp_{data_id}",
            )
            logger.info(f"MP subscription activated: {telegram_id} -> {plan}")

    return JSONResponse({"received": True})


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
