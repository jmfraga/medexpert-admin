"""
MedExpert LLM Benchmark — Compare 4 models with same RAG context.

Runs 5 clinical test cases against Claude Opus 4.6, GPT-5.2, GPT-OSS 20B (Groq),
and Llama 3.1 8B (Groq). Same ChromaDB RAG retrieval for all models.

Usage:
    python llm_benchmark.py                  # Run all tests
    python llm_benchmark.py --case 0         # Run single test case
    python llm_benchmark.py --models opus gpt5  # Run specific models
    python llm_benchmark.py --no-pdf         # Skip PDF generation

Output:
    data/benchmarks/benchmark_YYYYMMDD_HHMMSS.json
    data/benchmarks/benchmark_YYYYMMDD_HHMMSS.pdf  (landscape)
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv(override=True)

import database as db
from rag_engine import get_rag_for_expert
from bot_brain import (
    _translate_query_to_english, _diversify_results, _detect_society,
    DEFAULT_SYSTEM_PROMPT,
)

logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("benchmark")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# ─────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────

MODELS = {
    "opus": {
        "name": "Claude Opus 4.6",
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "color": (0.55, 0.27, 0.07),  # brown for PDF
    },
    "gpt5": {
        "name": "GPT-5.2",
        "provider": "openai",
        "model_id": "gpt-5.2",
        "api_key_env": "OPENAI_API_KEY",
        "color": (0.10, 0.46, 0.25),  # green
    },
    "gptoss": {
        "name": "GPT-OSS 20B (Groq)",
        "provider": "groq",
        "model_id": "openai/gpt-oss-20b",
        "api_key_env": "GROQ_API_KEY",
        "color": (0.13, 0.27, 0.53),  # blue
    },
    "gptoss120": {
        "name": "GPT-OSS 120B (Groq)",
        "provider": "groq",
        "model_id": "openai/gpt-oss-120b",
        "api_key_env": "GROQ_API_KEY",
        "color": (0.40, 0.15, 0.55),  # purple
    },
    "llama8b": {
        "name": "Llama 3.1 8B (Groq)",
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "api_key_env": "GROQ_API_KEY",
        "color": (0.55, 0.10, 0.10),  # red
    },
}

# ─────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "lung_iiia",
        "title": "Cáncer de pulmón estadio IIIA",
        "query": (
            "Paciente masculino de 58 años diagnosticado con cáncer de pulmón "
            "no microcítico (NSCLC) estadio IIIA, adenocarcinoma, EGFR negativo, "
            "ALK negativo, PD-L1 50%. ¿Cuál es el esquema de quimioterapia "
            "recomendado y el plan de tratamiento según las guías vigentes?"
        ),
    },
    {
        "id": "melanoma_pembro",
        "title": "Pembrolizumab en melanoma metastásico",
        "query": (
            "Paciente femenino de 45 años con melanoma cutáneo metastásico, "
            "BRAF V600E positivo, sin metástasis cerebrales. ¿Cuáles son las "
            "indicaciones de pembrolizumab según NCCN y ESMO? ¿Se recomienda "
            "como primera línea o después de terapia dirigida con BRAF/MEK?"
        ),
    },
    {
        "id": "ovarian_carbo",
        "title": "Carboplatino + Paclitaxel en cáncer de ovario",
        "query": (
            "Paciente femenino de 62 años con cáncer de ovario epitelial seroso "
            "de alto grado, estadio IIIC, post-cirugía de citorreducción óptima. "
            "¿Cuál es la dosis estándar de carboplatino + paclitaxel? ¿Cada cuánto "
            "se administra y cuántos ciclos? ¿Se recomienda bevacizumab de mantenimiento?"
        ),
    },
    {
        "id": "febrile_neutropenia",
        "title": "Neutropenia febril: manejo",
        "query": (
            "Paciente masculino de 70 años en quimioterapia con FOLFOX por cáncer "
            "de colon estadio III, llega a urgencias con fiebre de 38.5°C y "
            "neutrófilos en 400/mm³. ¿Cuál es el manejo inicial recomendado? "
            "¿Criterios de hospitalización vs ambulatorio? ¿Antibióticos empíricos?"
        ),
    },
    {
        "id": "folfox_vs_folfiri",
        "title": "FOLFOX vs FOLFIRI: comparación",
        "query": (
            "¿Cuáles son las diferencias entre FOLFOX y FOLFIRI como primera línea "
            "en cáncer colorrectal metastásico? Comparar: eficacia, toxicidad, "
            "indicaciones específicas, y preferencia según perfil del paciente. "
            "¿Cuándo se prefiere uno sobre el otro según NCCN y ESMO?"
        ),
    },
]

# ─────────────────────────────────────────────
# LLM Client Setup
# ─────────────────────────────────────────────

def _get_client(model_cfg: dict):
    """Create an LLM client for the given model config."""
    provider = model_cfg["provider"]

    if provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.getenv(model_cfg["api_key_env"]))

    elif provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv(model_cfg["api_key_env"]))

    elif provider == "groq":
        from openai import OpenAI
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv(model_cfg["api_key_env"]),
        )


def _call_model(client, model_cfg: dict, system: str, user_msg: str,
                max_tokens: int = 1500) -> dict:
    """Call an LLM and return response + metrics."""
    provider = model_cfg["provider"]
    model_id = model_cfg["model_id"]
    start = time.time()

    try:
        reasoning_tokens = 0

        if provider == "anthropic":
            resp = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
                timeout=90.0,
            )
            text = resp.content[0].text
            tokens_in = resp.usage.input_tokens if resp.usage else 0
            tokens_out = resp.usage.output_tokens if resp.usage else 0

        else:
            # OpenAI-compatible (OpenAI, Groq)
            # GPT-5+ requires max_completion_tokens (includes reasoning tokens)
            # Need higher limit for reasoning models since reasoning tokens
            # count against the limit
            if "gpt-5" in model_id or "gpt-4o" in model_id:
                token_param = {"max_completion_tokens": max_tokens * 6}
            else:
                token_param = {"max_tokens": max_tokens}
            resp = client.chat.completions.create(
                model=model_id,
                **token_param,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                timeout=90.0,
            )
            msg = resp.choices[0].message
            text = msg.content or ""
            # GPT-5+ reasoning models may return empty content — check for
            # reasoning summary or refusal fields
            if not text and hasattr(msg, "refusal") and msg.refusal:
                text = f"[REFUSAL] {msg.refusal}"
            tokens_in = getattr(resp.usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(resp.usage, "completion_tokens", 0) or 0
            # Include reasoning tokens if available
            reasoning_tokens = 0
            if hasattr(resp.usage, "completion_tokens_details"):
                details = resp.usage.completion_tokens_details
                reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        elapsed = time.time() - start
        result = {
            "status": "success",
            "response": text,
            "tokens_input": tokens_in,
            "tokens_output": tokens_out,
            "response_time": round(elapsed, 2),
            "error": None,
        }
        if provider in ("openai", "groq") and reasoning_tokens:
            result["reasoning_tokens"] = reasoning_tokens
        return result

    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"{model_cfg['name']} failed: {e}")
        return {
            "status": "error",
            "response": f"ERROR: {e}",
            "tokens_input": 0,
            "tokens_output": 0,
            "response_time": round(elapsed, 2),
            "error": str(e),
        }


# ─────────────────────────────────────────────
# RAG Context (shared across all models)
# ─────────────────────────────────────────────

def build_rag_context(query: str, expert_slug: str = "oncologia") -> tuple[str, list[str]]:
    """Build RAG context using bilingual search. Returns (context, citations)."""
    rag = get_rag_for_expert(expert_slug)

    # Bilingual search
    rag_es = rag.search_detailed(query, n_results=15)
    english_query = _translate_query_to_english(query)
    rag_en = rag.search_detailed(english_query, n_results=15) if english_query else []

    # Merge & dedup
    seen = set()
    merged = []
    for hit in rag_es + rag_en:
        key = (hit.get("source", ""), hit.get("text", "")[:100])
        if key not in seen:
            seen.add(key)
            merged.append(hit)

    diverse_hits = _diversify_results(merged, max_per_source=3, total=10)

    # Build context string
    context_parts = []
    citations = []
    for hit in diverse_hits:
        source = hit.get("source", "Guia clinica")
        section = hit.get("section_path", "")
        society = hit.get("society", "") or _detect_society(source)
        label = source
        if section:
            label += f" > {section}"
        if society:
            label = f"[{society}] {label}"
        context_parts.append(f"[{label}]: {hit['text'][:500]}")
        if label not in citations:
            citations.append(label)

    context = "\n\n".join(context_parts) if context_parts else "(Sin guias disponibles)"
    return context, citations


# ─────────────────────────────────────────────
# Benchmark Runner
# ─────────────────────────────────────────────

def run_benchmark(case_indices: list[int] = None, model_keys: list[str] = None,
                  parallel: bool = True) -> dict:
    """Run benchmark across test cases and models."""
    db.init_db()

    cases = TEST_CASES if case_indices is None else [TEST_CASES[i] for i in case_indices]
    models_to_run = model_keys or list(MODELS.keys())

    # Validate API keys
    for mk in models_to_run:
        cfg = MODELS[mk]
        key = os.getenv(cfg["api_key_env"])
        if not key:
            logger.error(f"Missing API key: {cfg['api_key_env']} for {cfg['name']}")
            sys.exit(1)

    # Initialize clients once
    clients = {}
    for mk in models_to_run:
        cfg = MODELS[mk]
        clients[mk] = _get_client(cfg)
        logger.info(f"Client ready: {cfg['name']}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "models": {mk: MODELS[mk]["name"] for mk in models_to_run},
        "cases": [],
    }

    for case in cases:
        print(f"\n{'='*80}")
        print(f"CASO: {case['title']}")
        print(f"{'='*80}")
        print(f"Query: {case['query'][:100]}...")

        # RAG retrieval (shared)
        rag_start = time.time()
        rag_context, citations = build_rag_context(case["query"])
        rag_time = round(time.time() - rag_start, 2)
        print(f"RAG: {len(citations)} citations ({rag_time}s)")

        # Build prompts
        system = DEFAULT_SYSTEM_PROMPT.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            rag_context=rag_context,
        )
        user_msg = (
            f"Consulta clinica:\n\n{case['query']}\n\n"
            "FORMATO DE RESPUESTA:\n"
            "- Responde en español médico profesional\n"
            "- Usa MAYÚSCULAS para títulos de sección\n"
            "- Usa viñetas con • para listas\n"
            "- Cita fuentes entre corchetes: [NCCN 2024], [ESMO]\n"
            "- Menciona datos clínicos relevantes del caso\n"
            "- Da una respuesta completa y detallada\n"
            "- Termina con CONSIDERACIONES ADICIONALES si aplica"
        )

        case_result = {
            "id": case["id"],
            "title": case["title"],
            "query": case["query"],
            "rag_time": rag_time,
            "citations": citations,
            "rag_context_length": len(rag_context),
            "responses": {},
        }

        # Call models
        if parallel:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for mk in models_to_run:
                    cfg = MODELS[mk]
                    fut = executor.submit(
                        _call_model, clients[mk], cfg, system, user_msg
                    )
                    futures[fut] = mk

                for fut in as_completed(futures):
                    mk = futures[fut]
                    res = fut.result()
                    case_result["responses"][mk] = res
                    status = "OK" if res["status"] == "success" else "ERROR"
                    print(f"  {MODELS[mk]['name']:25s} {status} {res['response_time']:.1f}s "
                          f"({res['tokens_input']}in/{res['tokens_output']}out)")
        else:
            for mk in models_to_run:
                cfg = MODELS[mk]
                print(f"  Calling {cfg['name']}...", end=" ", flush=True)
                res = _call_model(clients[mk], cfg, system, user_msg)
                case_result["responses"][mk] = res
                status = "OK" if res["status"] == "success" else "ERROR"
                print(f"{status} {res['response_time']:.1f}s "
                      f"({res['tokens_input']}in/{res['tokens_output']}out)")

        results["cases"].append(case_result)

        # Print responses preview
        for mk in models_to_run:
            r = case_result["responses"].get(mk, {})
            resp_text = r.get("response", "N/A")
            print(f"\n  --- {MODELS[mk]['name']} ({r.get('response_time', 0):.1f}s) ---")
            # Show first 300 chars
            preview = resp_text[:300].replace("\n", "\n  ")
            print(f"  {preview}...")

    # Summary table
    _print_summary(results, models_to_run)

    return results


def _print_summary(results: dict, model_keys: list[str]):
    """Print a summary table in terminal."""
    print(f"\n\n{'='*80}")
    print("RESUMEN DEL BENCHMARK")
    print(f"{'='*80}")

    # Header
    header = f"{'Caso':<35s}"
    for mk in model_keys:
        header += f" {MODELS[mk]['name']:>18s}"
    print(header)
    print("-" * len(header))

    # Time per case
    for case in results["cases"]:
        row = f"{case['title']:<35s}"
        for mk in model_keys:
            r = case["responses"].get(mk, {})
            t = r.get("response_time", 0)
            status = "" if r.get("status") == "success" else " ERR"
            row += f" {t:>15.1f}s{status}"
        print(row)

    # Averages
    print("-" * len(header))
    avg_row = f"{'PROMEDIO':<35s}"
    for mk in model_keys:
        times = [c["responses"].get(mk, {}).get("response_time", 0)
                 for c in results["cases"]
                 if c["responses"].get(mk, {}).get("status") == "success"]
        avg = sum(times) / len(times) if times else 0
        avg_row += f" {avg:>15.1f}s"
    print(avg_row)

    # Total tokens
    tok_row = f"{'TOKENS TOTALES (in/out)':<35s}"
    for mk in model_keys:
        tin = sum(c["responses"].get(mk, {}).get("tokens_input", 0) for c in results["cases"])
        tout = sum(c["responses"].get(mk, {}).get("tokens_output", 0) for c in results["cases"])
        tok_row += f" {tin:>7d}/{tout:<7d}  "
    print(tok_row)

    # Response length avg
    len_row = f"{'LONG. PROMEDIO RESPUESTA':<35s}"
    for mk in model_keys:
        lengths = [len(c["responses"].get(mk, {}).get("response", ""))
                   for c in results["cases"]
                   if c["responses"].get(mk, {}).get("status") == "success"]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        len_row += f" {avg_len:>14.0f}ch  "
    print(len_row)


# ─────────────────────────────────────────────
# PDF Export (Landscape)
# ─────────────────────────────────────────────

def export_pdf(results: dict, pdf_path: str, model_keys: list[str] = None):
    """Export benchmark results to a landscape PDF."""
    try:
        import fitz
    except ImportError:
        logger.error("PyMuPDF not installed, skipping PDF")
        return

    model_keys = model_keys or list(results["models"].keys())
    num_models = len(model_keys)

    # Landscape Letter
    page_w, page_h = 842, 595  # A4 landscape
    margin = 40
    content_w = page_w - 2 * margin

    doc = fitz.open()

    # ── Title page ──
    page = doc.new_page(width=page_w, height=page_h)
    # Header bar
    page.draw_rect(fitz.Rect(0, 0, page_w, 50), color=None, fill=(0.12, 0.25, 0.50))
    page.insert_text(fitz.Point(margin, 35), "MedExpert LLM Benchmark",
                     fontsize=22, fontname="hebo", color=(1, 1, 1))
    page.insert_text(fitz.Point(page_w - margin - 200, 35),
                     datetime.now().strftime("%d/%m/%Y %H:%M"),
                     fontsize=10, fontname="helv", color=(0.8, 0.85, 0.95))

    y = 75
    # Models table
    page.insert_text(fitz.Point(margin, y), "MODELOS EVALUADOS:",
                     fontsize=11, fontname="hebo", color=(0.12, 0.25, 0.50))
    y += 18
    for mk in model_keys:
        cfg = MODELS.get(mk, {})
        page.insert_text(fitz.Point(margin + 10, y),
                         f"• {cfg.get('name', mk)} ({cfg.get('model_id', '?')})",
                         fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 14

    y += 10
    page.insert_text(fitz.Point(margin, y), "CASOS DE PRUEBA:",
                     fontsize=11, fontname="hebo", color=(0.12, 0.25, 0.50))
    y += 18
    for case in results["cases"]:
        page.insert_text(fitz.Point(margin + 10, y),
                         f"• {case['title']}",
                         fontsize=9, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 14

    # Summary metrics table
    y += 15
    page.insert_text(fitz.Point(margin, y), "RESUMEN DE MÉTRICAS:",
                     fontsize=11, fontname="hebo", color=(0.12, 0.25, 0.50))
    y += 20

    # Table header
    col_w = content_w / (num_models + 1)
    page.insert_text(fitz.Point(margin, y), "Métrica",
                     fontsize=8, fontname="hebo", color=(0.3, 0.3, 0.3))
    for i, mk in enumerate(model_keys):
        x = margin + col_w * (i + 1)
        cfg = MODELS.get(mk, {})
        page.insert_text(fitz.Point(x, y), cfg.get("name", mk),
                         fontsize=8, fontname="hebo", color=cfg.get("color", (0, 0, 0)))
    y += 4
    page.draw_line(fitz.Point(margin, y), fitz.Point(page_w - margin, y),
                   color=(0.7, 0.7, 0.7), width=0.5)
    y += 12

    # Rows: avg time, total tokens, avg response length
    metrics = []

    # Avg time
    avg_times = {}
    for mk in model_keys:
        times = [c["responses"].get(mk, {}).get("response_time", 0)
                 for c in results["cases"]
                 if c["responses"].get(mk, {}).get("status") == "success"]
        avg_times[mk] = f"{sum(times)/len(times):.1f}s" if times else "N/A"
    metrics.append(("Tiempo promedio", avg_times))

    # Total tokens
    tok_vals = {}
    for mk in model_keys:
        tin = sum(c["responses"].get(mk, {}).get("tokens_input", 0) for c in results["cases"])
        tout = sum(c["responses"].get(mk, {}).get("tokens_output", 0) for c in results["cases"])
        tok_vals[mk] = f"{tin:,}in / {tout:,}out"
    metrics.append(("Tokens totales", tok_vals))

    # Avg response length
    len_vals = {}
    for mk in model_keys:
        lengths = [len(c["responses"].get(mk, {}).get("response", ""))
                   for c in results["cases"]
                   if c["responses"].get(mk, {}).get("status") == "success"]
        avg_l = sum(lengths) / len(lengths) if lengths else 0
        len_vals[mk] = f"{avg_l:,.0f} chars"
    metrics.append(("Long. promedio respuesta", len_vals))

    # Errors
    err_vals = {}
    for mk in model_keys:
        errs = sum(1 for c in results["cases"]
                   if c["responses"].get(mk, {}).get("status") == "error")
        err_vals[mk] = str(errs)
    metrics.append(("Errores", err_vals))

    for label, vals in metrics:
        page.insert_text(fitz.Point(margin, y), label,
                         fontsize=8, fontname="helv", color=(0.2, 0.2, 0.2))
        for i, mk in enumerate(model_keys):
            x = margin + col_w * (i + 1)
            page.insert_text(fitz.Point(x, y), vals.get(mk, "N/A"),
                             fontsize=8, fontname="helv", color=(0.15, 0.15, 0.15))
        y += 14

    # Per-case time breakdown
    y += 10
    page.insert_text(fitz.Point(margin, y), "Caso",
                     fontsize=8, fontname="hebo", color=(0.3, 0.3, 0.3))
    for i, mk in enumerate(model_keys):
        x = margin + col_w * (i + 1)
        cfg = MODELS.get(mk, {})
        page.insert_text(fitz.Point(x, y), cfg.get("name", mk),
                         fontsize=8, fontname="hebo", color=cfg.get("color", (0, 0, 0)))
    y += 4
    page.draw_line(fitz.Point(margin, y), fitz.Point(page_w - margin, y),
                   color=(0.7, 0.7, 0.7), width=0.5)
    y += 12

    for case in results["cases"]:
        title = case["title"][:40]
        page.insert_text(fitz.Point(margin, y), title,
                         fontsize=8, fontname="helv", color=(0.2, 0.2, 0.2))
        for i, mk in enumerate(model_keys):
            x = margin + col_w * (i + 1)
            r = case["responses"].get(mk, {})
            t = r.get("response_time", 0)
            status = f"{t:.1f}s" if r.get("status") == "success" else "ERROR"
            page.insert_text(fitz.Point(x, y), status,
                             fontsize=8, fontname="helv", color=(0.15, 0.15, 0.15))
        y += 14

    # ── Per-case detail pages (full responses, one model per page) ──
    line_h = 9
    font_size = 7
    max_y = page_h - 45  # leave room for footer

    def _new_detail_page(doc, title_text, subtitle_text=None):
        """Create a new detail page with header bar."""
        pg = doc.new_page(width=page_w, height=page_h)
        pg.draw_rect(fitz.Rect(0, 0, page_w, 40), color=None, fill=(0.12, 0.25, 0.50))
        pg.insert_text(fitz.Point(margin, 28), title_text,
                       fontsize=14, fontname="hebo", color=(1, 1, 1))
        if subtitle_text:
            pg.insert_text(fitz.Point(page_w - margin - 250, 28), subtitle_text,
                           fontsize=9, fontname="helv", color=(0.8, 0.85, 0.95))
        return pg, 55

    for case in results["cases"]:
        # Case intro page with query + citations
        page, y = _new_detail_page(doc, case["title"])

        # Query
        page.insert_text(fitz.Point(margin, y), "CONSULTA:",
                         fontsize=9, fontname="hebo", color=(0.3, 0.3, 0.3))
        y += 14
        query_lines = _wrap_text_pdf(case["query"], int(content_w / (8 * 0.52)))
        for line in query_lines:
            page.insert_text(fitz.Point(margin + 5, y), line,
                             fontsize=8, fontname="helv", color=(0.2, 0.2, 0.2))
            y += 11
        y += 8

        # Citations
        page.insert_text(fitz.Point(margin, y),
                         f"RAG ({case['rag_time']}s, {len(case['citations'])} refs):",
                         fontsize=8, fontname="hebo", color=(0.3, 0.3, 0.3))
        y += 12
        for cite in case["citations"][:8]:
            page.insert_text(fitz.Point(margin + 5, y), f"• {cite[:100]}",
                             fontsize=7, fontname="helv", color=(0.4, 0.4, 0.4))
            y += 10
        y += 10

        # Quick comparison table on intro page
        page.insert_text(fitz.Point(margin, y), "COMPARACIÓN RÁPIDA:",
                         fontsize=9, fontname="hebo", color=(0.12, 0.25, 0.50))
        y += 16
        for mk in model_keys:
            if mk not in case["responses"]:
                continue
            cfg = MODELS.get(mk, {})
            r = case["responses"][mk]
            status = "OK" if r.get("status") == "success" else "ERROR"
            line_text = (f"{cfg.get('name', mk):22s}  |  {r.get('response_time', 0):5.1f}s  |  "
                        f"{r.get('tokens_input', 0)}in/{r.get('tokens_output', 0)}out  |  "
                        f"{len(r.get('response', '')):,} chars  |  {status}")
            page.insert_text(fitz.Point(margin + 5, y), line_text,
                             fontsize=8, fontname="helv",
                             color=cfg.get("color", (0.2, 0.2, 0.2)))
            y += 14

        # Full response pages — one model per page (with continuation pages)
        for mk in model_keys:
            if mk not in case["responses"]:
                continue
            cfg = MODELS.get(mk, {})
            r = case["responses"][mk]
            resp_text = r.get("response", "ERROR")

            # Clean markdown for PDF
            clean = resp_text.replace("**", "").replace("*", "").replace("##", "").replace("#", "")
            lines = _wrap_text_pdf(clean, int(content_w / (font_size * 0.52)))

            # Metrics subtitle
            metrics_sub = (f"{r.get('response_time', 0):.1f}s | "
                          f"{r.get('tokens_input', 0)}in/{r.get('tokens_output', 0)}out | "
                          f"{len(resp_text):,} chars")

            page, y = _new_detail_page(doc, case["title"], cfg.get("name", mk))

            # Model name + metrics
            page.insert_text(fitz.Point(margin, y), cfg.get("name", mk),
                             fontsize=12, fontname="hebo",
                             color=cfg.get("color", (0, 0, 0)))
            y += 15
            page.insert_text(fitz.Point(margin, y), metrics_sub,
                             fontsize=8, fontname="helv", color=(0.4, 0.4, 0.4))
            y += 5
            page.draw_line(fitz.Point(margin, y),
                          fitz.Point(page_w - margin, y),
                          color=cfg.get("color", (0.5, 0.5, 0.5)), width=1)
            y += 10

            # Write all lines with automatic page breaks
            page_num = 1
            for line in lines:
                if y + line_h > max_y:
                    page_num += 1
                    page, y = _new_detail_page(
                        doc, case["title"],
                        f"{cfg.get('name', mk)} (pág. {page_num})"
                    )
                    y += 5

                page.insert_text(fitz.Point(margin, y), line,
                                 fontsize=font_size, fontname="helv",
                                 color=(0.1, 0.1, 0.1))
                y += line_h

    # Save
    doc.save(pdf_path)
    doc.close()
    logger.info(f"PDF exported: {pdf_path}")


def _wrap_text_pdf(text: str, max_chars: int) -> list[str]:
    """Wrap text for PDF columns."""
    all_lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            all_lines.append("")
            continue
        words = paragraph.split()
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            if len(test) > max_chars:
                if current:
                    all_lines.append(current)
                current = word
            else:
                current = test
        if current:
            all_lines.append(current)
    return all_lines or [""]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MedExpert LLM Benchmark")
    parser.add_argument("--case", type=int, default=None,
                        help="Run single test case by index (0-4)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODELS.keys()),
                        help="Models to test (default: all)")
    parser.add_argument("--sequential", action="store_true",
                        help="Run models sequentially instead of parallel")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Skip PDF generation")
    args = parser.parse_args()

    case_indices = [args.case] if args.case is not None else None
    model_keys = args.models or list(MODELS.keys())

    print(f"\nMedExpert LLM Benchmark")
    print(f"Models: {', '.join(MODELS[mk]['name'] for mk in model_keys)}")
    cases_desc = f"Case {args.case}" if args.case is not None else f"All {len(TEST_CASES)} cases"
    print(f"Cases: {cases_desc}")
    print(f"Mode: {'sequential' if args.sequential else 'parallel'}")

    # Run
    results = run_benchmark(
        case_indices=case_indices,
        model_keys=model_keys,
        parallel=not args.sequential,
    )

    # Save results
    out_dir = Path("data/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = str(out_dir / f"benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON saved: {json_path}")

    if not args.no_pdf:
        pdf_path = str(out_dir / f"benchmark_{timestamp}.pdf")
        export_pdf(results, pdf_path, model_keys)
        print(f"PDF saved: {pdf_path}")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
