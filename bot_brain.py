"""
MedExpert Telegram Bot - Clinical Brain
LLM consultation engine with RAG for the Telegram bot.
Adapted from medexpert-client clinical_brain.py for single-query pattern.
"""

import os
import time
import tempfile
import logging
from datetime import datetime
from pathlib import Path

from rag_engine import get_rag_for_expert
import database as db

logger = logging.getLogger(__name__)

# Default system prompt when expert has none configured
DEFAULT_SYSTEM_PROMPT = """Eres MedExpert, un asistente clinico especializado.
Fecha: {current_date}

INSTRUCCIONES:
- Responde en espanol medico profesional
- Basa tus respuestas en las guias clinicas proporcionadas
- Cita las fuentes cuando sea posible (ej: [NCCN 2024], [ESMO])
- Si no tienes informacion suficiente, indicalo claramente
- NO inventes datos clinicos ni estadisticas
- Recuerda: eres una herramienta de APOYO, no reemplazas el criterio medico

ESTRUCTURA DE RESPUESTA (formato SAER - obligatorio):
Usa la nemotecnia SAER para estructurar TODAS tus respuestas:

1. SITUACION: Resumen breve del caso clinico o pregunta del medico
2. ANTECEDENTES: Contexto clinico relevante (epidemiologia, fisiopatologia, factores de riesgo)
3. EVALUACION: Analisis basado en las guias clinicas con niveles de evidencia
4. RECOMENDACIONES: Conducta sugerida, opciones terapeuticas, seguimiento

Al final incluye:
- GUIAS CONSULTADAS: lista de guias citadas con año
- Si aplica: REFERENCIAS PUBMED

GUIAS CLINICAS DISPONIBLES:
{rag_context}"""

# Whisper singleton (lazy-loaded)
_whisper_model = None


def get_whisper():
    """Lazy-load Whisper model (expensive, only when first voice message arrives)."""
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        model_size = os.getenv("BOT_WHISPER_MODEL", "medium")
        logger.info(f"Loading Whisper {model_size}...")
        _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        logger.info(f"Whisper {model_size} ready")
        return _whisper_model
    except ImportError:
        logger.error("faster-whisper not installed. Voice messages won't work.")
        return None
    except Exception as e:
        logger.error(f"Whisper init failed: {e}")
        return None


async def transcribe_audio(audio_path: str, expert_slug: str = "") -> str | None:
    """Transcribe audio file using Whisper. Returns text or None on failure."""
    model = get_whisper()
    if model is None:
        return None

    try:
        # Build initial_prompt from glossary (includes synonyms/brand names)
        initial_prompt = "Sesion clinica."
        if expert_slug:
            terms = db.get_glossary_terms_for_expert_by_slug(expert_slug)
            if terms:
                all_names = []
                for t in terms[:100]:
                    all_names.append(t["term"])
                    if t.get("synonyms"):
                        all_names.extend(s.strip() for s in t["synonyms"].split(",") if s.strip())
                initial_prompt = f"Sesion clinica. {', '.join(all_names)}."

        language = os.getenv("BOT_WHISPER_LANGUAGE", "es")
        segments, info = model.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            initial_prompt=initial_prompt,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        logger.info(f"Transcribed {info.duration:.1f}s audio -> {len(text)} chars")
        return text.strip() if text.strip() else None
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


def _expand_synonyms(text: str, expert_slug: str) -> str:
    """Expand brand names to generic names (and vice versa) in query text.
    E.g. if user says 'Keytruda', also search for 'pembrolizumab'."""
    terms = db.get_glossary_terms_for_expert_by_slug(expert_slug)
    if not terms:
        return text
    text_lower = text.lower()
    expansions = []
    for t in terms:
        synonyms_str = t.get("synonyms", "")
        if not synonyms_str:
            continue
        syn_list = [s.strip() for s in synonyms_str.split(",") if s.strip()]
        all_names = [t["term"]] + syn_list
        for name in all_names:
            if name.lower() in text_lower:
                # Add all other names as expansions
                for other in all_names:
                    if other.lower() != name.lower() and other.lower() not in text_lower:
                        expansions.append(other)
    if expansions:
        return f"{text} ({', '.join(expansions)})"
    return text


class BotBrain:
    """Clinical consultation engine for Telegram bot queries."""

    def __init__(self, provider: str = None, model: str = None,
                 deepen_provider: str = None, deepen_model: str = None,
                 deepen_premium_provider: str = None, deepen_premium_model: str = None):
        settings = db.get_all_settings()
        self.provider = provider or settings.get("default_provider", "anthropic")
        self.model = model or settings.get("default_model", "claude-haiku-4-5-20251001")
        self.deepen_provider = deepen_provider or settings.get("default_deepen_provider", "anthropic")
        self.deepen_model = deepen_model or settings.get("default_deepen_model", "claude-sonnet-4-20250514")
        self.deepen_premium_provider = deepen_premium_provider or settings.get("default_deepen_premium_provider", "anthropic")
        self.deepen_premium_model = deepen_premium_model or settings.get("default_deepen_premium_model", "claude-opus-4-6")
        # Fallback chain (configurable from admin)
        self.fallback_chain = []
        fb1_provider = settings.get("fallback1_provider", "")
        fb1_model = settings.get("fallback1_model", "")
        if fb1_provider and fb1_model:
            self.fallback_chain.append((fb1_provider, fb1_model))
        fb2_provider = settings.get("fallback2_provider", "")
        fb2_model = settings.get("fallback2_model", "")
        if fb2_provider and fb2_model:
            self.fallback_chain.append((fb2_provider, fb2_model))
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            if self.provider == "anthropic":
                from anthropic import Anthropic
                self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                logger.info(f"Anthropic ready (model: {self.model})")
            elif self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info(f"OpenAI ready (model: {self.model})")
            elif self.provider == "groq":
                from openai import OpenAI
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=os.getenv("GROQ_API_KEY"),
                )
                logger.info(f"Groq ready (model: {self.model})")
            elif self.provider == "synapse":
                from openai import OpenAI
                synapse_url = os.getenv("SYNAPSE_BASE_URL", "http://100.72.169.113:8800/v1")
                self.client = OpenAI(base_url=synapse_url, api_key=os.getenv("SYNAPSE_API_KEY"))
                logger.info(f"Synapse ready (model: {self.model}, url: {synapse_url})")
            elif self.provider == "ollama":
                from openai import OpenAI
                ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
                self.client = OpenAI(base_url=ollama_url, api_key="ollama")
                logger.info(f"Ollama ready (model: {self.model})")
            else:
                logger.error(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"LLM client init error: {e}")

    def query(self, text: str, expert_slug: str,
              source_filter: dict | None = None,
              tier: str = "free") -> dict:
        """Run a clinical query: RAG search + LLM reasoning.

        Returns dict with: status, response, rag_context, provider, model,
                          token_usage, processing_time, citations
        """
        start = time.time()

        # Get expert info for system prompt
        expert = db.get_expert_by_slug(expert_slug)
        system_prompt = (expert["system_prompt"] if expert and expert.get("system_prompt")
                         else DEFAULT_SYSTEM_PROMPT)

        # Expand brand names → generic names (e.g. Keytruda → pembrolizumab)
        expanded_text = _expand_synonyms(text, expert_slug)

        # RAG search — dual-language: Spanish (original) + English (translated)
        # NCCN/ESMO guidelines are in English, IMSS in Spanish.
        # Without bilingual search, cosine similarity always favors same-language chunks.
        rag = get_rag_for_expert(expert_slug)
        rag_es = rag.search_detailed(expanded_text, n_results=15, where=source_filter)
        english_query = _translate_query_to_english(expanded_text)
        rag_en = rag.search_detailed(english_query, n_results=15, where=source_filter) if english_query else []
        logger.info(f"RAG: {len(rag_es)} ES hits + {len(rag_en)} EN hits")

        # Merge results, deduplicate by (source + text[:100])
        seen = set()
        merged = []
        for hit in rag_es + rag_en:
            key = (hit.get("source", ""), hit.get("text", "")[:100])
            if key not in seen:
                seen.add(key)
                merged.append(hit)

        diverse_hits = _diversify_results(merged, max_per_source=3, total=10)

        # Build context from diverse hits
        context_parts = []
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
        rag_context = "\n\n".join(context_parts) if context_parts else "(Sin guias disponibles)"

        # Build system prompt with context
        try:
            system = system_prompt.format(
                current_date=datetime.now().strftime("%Y-%m-%d"),
                rag_context=rag_context,
            )
        except (KeyError, IndexError):
            system = system_prompt + f"\n\nFecha: {datetime.now().strftime('%Y-%m-%d')}\n\nGUIAS CLINICAS:\n{rag_context}"

        # User message — SAER structure + Telegram-friendly formatting
        user_message = (
            "Consulta clinica:\n\n"
            f"{text}\n\n"
            "ESTRUCTURA SAER (OBLIGATORIO):\n"
            "Organiza tu respuesta asi:\n\n"
            "SITUACION:\n"
            "Resumen del caso o pregunta con datos clinicos relevantes (edad, diagnostico, estadio, etc.)\n\n"
            "ANTECEDENTES:\n"
            "Contexto clinico: epidemiologia, fisiopatologia, clasificacion, factores pronosticos relevantes\n\n"
            "EVALUACION:\n"
            "Analisis basado en guias clinicas. Incluye niveles de evidencia y grados de recomendacion. "
            "Cita fuentes: [NCCN 2024], [ESMO], [CMCM 2025]\n\n"
            "RECOMENDACIONES:\n"
            "Conducta sugerida: opciones terapeuticas, dosis si aplica, seguimiento, referencia a especialista\n\n"
            "GUIAS CONSULTADAS:\n"
            "Lista de guias citadas con año\n\n"
            "FORMATO (OBLIGATORIO - texto plano para Telegram):\n"
            "- PROHIBIDO usar # ## ### para encabezados\n"
            "- PROHIBIDO usar tablas con | o ---\n"
            "- PROHIBIDO usar ** para negritas ni ningun markdown\n"
            "- Para titulos de seccion: escribe en MAYUSCULAS seguido de dos puntos\n"
            "- Para listas: usa viñetas con • al inicio de cada punto\n"
            "- Para sub-listas: usa guion - con indentacion\n"
            "- Separa secciones con una linea en blanco"
        )

        # Call LLM — extended thinking for Haiku gives near-Sonnet quality
        result = self._call_llm(system, user_message, max_tokens=2000, extended_thinking=True)

        # Fallback chain if primary fails
        if result["status"] == "error" and self.fallback_chain:
            original_provider = self.provider
            original_model = self.model
            for fb_provider, fb_model in self.fallback_chain:
                logger.warning(f"{self.provider}/{self.model} failed, trying fallback {fb_provider}/{fb_model}")
                self.provider = fb_provider
                self.model = fb_model
                self._init_client()
                result = self._call_llm(system, user_message, max_tokens=2000, extended_thinking=True)
                if result["status"] == "success":
                    result["fallback"] = True
                    result["original_provider"] = original_provider
                    break
            # Restore original for next query
            self.provider = original_provider
            self.model = original_model
            self._init_client()

        elapsed = time.time() - start
        result["processing_time"] = elapsed
        result["rag_context"] = rag_context
        result["rag_chunks_used"] = len(merged)
        result["provider"] = self.provider
        result["model"] = self.model

        # Extract citations from the diverse hits actually used in context
        citations = []
        for hit in diverse_hits:
            source = hit.get("source", "")
            society = hit.get("society", "") or _detect_society(source)
            section = hit.get("section_path", "")
            label = f"[{society}] {source}" if society else source
            if section:
                label += f" > {section}"
            if label not in citations:
                citations.append(label)
        result["citations"] = citations

        # Literature search for premium users on first response
        reference_papers = []
        if tier == "premium":
            reference_papers = _search_literature(text)
        result["pubmed_papers"] = reference_papers
        # Flag for Telegram formatting (don't inline, suggest PDF)
        result["has_references"] = len(reference_papers) > 0

        return result

    def _build_client(self, provider: str):
        """Build an LLM client for the given provider."""
        if provider == "anthropic":
            from anthropic import Anthropic
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif provider == "groq":
            from openai import OpenAI
            return OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
            )
        elif provider == "openai":
            from openai import OpenAI
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif provider == "synapse":
            from openai import OpenAI
            synapse_url = os.getenv("SYNAPSE_BASE_URL", "http://100.72.169.113:8800/v1")
            return OpenAI(base_url=synapse_url, api_key=os.getenv("SYNAPSE_API_KEY"))
        elif provider == "ollama":
            from openai import OpenAI
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            return OpenAI(base_url=ollama_url, api_key="ollama")
        else:
            logger.error(f"Unknown deepen provider: {provider}")
            return None

    def deepen(self, original_query: str, original_response: str,
               expert_slug: str, tier: str = "free",
               source_filter: dict | None = None,
               followup_question: str = None) -> dict:
        """Deepen a previous response with a more powerful model.

        Tiers:
          - "free": base model (self.provider / self.model)
          - "basic" / "premium": deepen model (self.deepen_provider / self.deepen_model)
        """
        start = time.time()

        # Select model based on tier
        if tier == "premium":
            deepen_provider = self.deepen_premium_provider
            deepen_model = self.deepen_premium_model
            client = self._build_client(deepen_provider)
            max_tokens = 3000
        elif tier == "basic":
            deepen_provider = self.deepen_provider
            deepen_model = self.deepen_model
            client = self._build_client(deepen_provider)
            max_tokens = 2500
        else:
            # Free: use base model
            deepen_provider = self.provider
            deepen_model = self.model
            client = self._build_client(deepen_provider)
            max_tokens = 2000

        # Re-do RAG search for fresh context (with synonym expansion)
        expanded_query = _expand_synonyms(original_query, expert_slug)
        rag = get_rag_for_expert(expert_slug)
        rag_es = rag.search_detailed(expanded_query, n_results=15, where=source_filter)
        english_query = _translate_query_to_english(expanded_query)
        rag_en = rag.search_detailed(english_query, n_results=15, where=source_filter) if english_query else []

        seen = set()
        merged = []
        for hit in rag_es + rag_en:
            key = (hit.get("source", ""), hit.get("text", "")[:100])
            if key not in seen:
                seen.add(key)
                merged.append(hit)

        diverse_hits = _diversify_results(merged, max_per_source=3, total=12)

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
            context_parts.append(f"[{label}]: {hit['text'][:600]}")
            if label not in citations:
                citations.append(label)

        rag_context = "\n\n".join(context_parts) if context_parts else "(Sin guias disponibles)"

        system = (
            "Eres MedExpert, asistente clinico especializado de alto nivel.\n"
            f"Fecha: {datetime.now().strftime('%Y-%m-%d')}\n\n"
            "Un medico solicita PROFUNDIZAR en una respuesta clinica previa.\n"
            "Tu rol es expandir significativamente usando la estructura SAER:\n\n"
            "ESTRUCTURA SAER para profundizacion:\n"
            "- SITUACION: Contexto actualizado del caso\n"
            "- ANTECEDENTES: Mecanismos de accion, farmacologia, estudios pivotales\n"
            "- EVALUACION: Evidencia detallada con niveles (IA, IB, IIA, etc.), "
            "dosis especificas, esquemas, duracion, efectos adversos y su manejo\n"
            "- RECOMENDACIONES: Algoritmo terapeutico, alternativas, criterios de seleccion, "
            "datos de supervivencia y eficacia\n\n"
            "Al final incluye:\n"
            "- GUIAS CONSULTADAS: con año y seccion especifica\n"
            "- Citas especificas: [NCCN 2024], [ESMO], etc.\n\n"
            f"GUIAS CLINICAS DISPONIBLES:\n{rag_context}"
        )

        if followup_question:
            user_message = (
                f"CONSULTA ORIGINAL:\n{original_query}\n\n"
                f"RESPUESTA PREVIA:\n{original_response}\n\n"
                f"PREGUNTA ESPECIFICA DEL MEDICO:\n{followup_question}\n\n"
                "INSTRUCCIONES:\n"
                "- Responde enfocandote en la pregunta especifica del medico\n"
                "- Usa la consulta original y respuesta previa como contexto\n"
                "- Estructura con SAER: SITUACION, ANTECEDENTES, EVALUACION, RECOMENDACIONES\n"
                "- Incluye niveles de evidencia y grados de recomendacion\n"
                "- Menciona estudios clinicos clave si los hay en las guias\n"
                "- Cita fuentes entre corchetes: [NCCN 2024], [ESMO], [CMCM 2025]\n"
                "- Responde en español medico profesional\n"
                "- Al final: GUIAS CONSULTADAS con año\n\n"
                "FORMATO (OBLIGATORIO - texto plano para Telegram):\n"
                "- PROHIBIDO usar # ## ### para encabezados\n"
                "- PROHIBIDO usar tablas con | o ---\n"
                "- PROHIBIDO usar ** para negritas ni ningun markdown\n"
                "- Titulos de seccion en MAYUSCULAS seguido de dos puntos\n"
                "- Listas con viñetas • y sub-listas con guion -\n"
                "- Separar secciones con linea en blanco"
            )
        else:
            user_message = (
                f"CONSULTA ORIGINAL:\n{original_query}\n\n"
                f"RESPUESTA PREVIA (a profundizar):\n{original_response}\n\n"
                "INSTRUCCIONES:\n"
                "- Profundiza y expande la respuesta anterior con mas detalle clinico\n"
                "- No repitas lo mismo, agrega informacion nueva y mas especifica\n"
                "- Estructura con SAER: SITUACION, ANTECEDENTES, EVALUACION, RECOMENDACIONES\n"
                "- Incluye niveles de evidencia y grados de recomendacion\n"
                "- Incluye dosis especificas, esquemas y duracion de tratamiento\n"
                "- Efectos adversos relevantes y su manejo\n"
                "- Alternativas terapeuticas y criterios de seleccion\n"
                "- Menciona estudios clinicos clave si los hay en las guias\n"
                "- Cita fuentes entre corchetes: [NCCN 2024], [ESMO], [CMCM 2025]\n"
                "- Responde en español medico profesional\n"
                "- Al final: GUIAS CONSULTADAS con año y seccion especifica\n\n"
                "FORMATO (OBLIGATORIO - texto plano para Telegram):\n"
                "- PROHIBIDO usar # ## ### para encabezados\n"
                "- PROHIBIDO usar tablas con | o ---\n"
                "- PROHIBIDO usar ** para negritas ni ningun markdown\n"
                "- Titulos de seccion en MAYUSCULAS seguido de dos puntos\n"
                "- Listas con viñetas • y sub-listas con guion -\n"
                "- Separar secciones con linea en blanco"
            )

        try:
            if deepen_provider == "anthropic":
                response = client.messages.create(
                    model=deepen_model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_message}],
                    timeout=90.0,
                )
                text = response.content[0].text
                token_usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            else:
                # OpenAI-compatible (Groq)
                response = client.chat.completions.create(
                    model=deepen_model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    timeout=90.0,
                )
                text = response.choices[0].message.content or ""
                token_usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                }

            logger.info(f"Deepen OK ({deepen_provider}/{deepen_model}) "
                        f"{token_usage['input_tokens']}in/{token_usage['output_tokens']}out")

            # Literature search for premium follow-up questions
            pubmed_papers = []
            if tier == "premium" and followup_question:
                pubmed_papers = _search_literature(followup_question)

            elapsed = time.time() - start
            return {
                "status": "success",
                "response": text,
                "token_usage": token_usage,
                "processing_time": elapsed,
                "provider": deepen_provider,
                "model": deepen_model,
                "rag_chunks_used": len(merged),
                "citations": citations,
                "pubmed_papers": pubmed_papers,
            }
        except Exception as e:
            logger.error(f"Deepen failed ({deepen_provider}/{deepen_model}): {e}")
            elapsed = time.time() - start
            return {
                "status": "error",
                "response": f"Error al profundizar: {e}",
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
                "processing_time": elapsed,
                "provider": deepen_provider,
                "model": deepen_model,
                "rag_chunks_used": 0,
                "citations": [],
            }

    def _call_llm(self, system: str, user_message: str, max_tokens: int = 800,
                  extended_thinking: bool = False) -> dict:
        if self.client is None:
            return {
                "status": "error",
                "response": "LLM no configurado. Verifica las API keys.",
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }

        try:
            token_usage = {"input_tokens": 0, "output_tokens": 0}

            if self.provider == "anthropic":
                # Extended thinking for Haiku — better quality at Haiku cost
                use_thinking = extended_thinking and "haiku" in self.model
                if use_thinking:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens + 8000,
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 8000,
                        },
                        system=system,
                        messages=[{"role": "user", "content": user_message}],
                        timeout=90.0,
                    )
                    # Extract text block (skip thinking blocks)
                    response_text = ""
                    for block in response.content:
                        if block.type == "text":
                            response_text = block.text
                            break
                else:
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user_message}],
                        timeout=45.0,
                    )
                    response_text = response.content[0].text
                if hasattr(response, "usage") and response.usage:
                    token_usage = {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
            else:
                # OpenAI-compatible (OpenAI, Groq, Ollama)
                is_gpt5 = "gpt-5" in self.model
                token_param = (
                    {"max_completion_tokens": max_tokens}
                    if is_gpt5
                    else {"max_tokens": max_tokens}
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    **token_param,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                    timeout=45.0,
                )
                response_text = response.choices[0].message.content
                if hasattr(response, "usage") and response.usage:
                    token_usage = {
                        "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                        "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                    }

            logger.info(
                f"LLM OK ({self.provider}/{self.model}) "
                f"{token_usage['input_tokens']}in/{token_usage['output_tokens']}out"
            )
            return {
                "status": "success",
                "response": response_text,
                "token_usage": token_usage,
            }

        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}/{self.model}): {e}")
            return {
                "status": "error",
                "response": f"Error: {e}",
                "error_type": type(e).__name__,
                "token_usage": {"input_tokens": 0, "output_tokens": 0},
            }


def _clean_markdown(text: str) -> str:
    """Strip markdown formatting that doesn't render well in Telegram plain text."""
    import re
    # Remove ### ## # headers — keep the text, strip the hashes
    text = re.sub(r'^#{1,4}\s*', '', text, flags=re.MULTILINE)
    # Remove **bold** and *italic* markers
    text = text.replace("**", "").replace("__", "")
    # Remove single * only if used as markdown italic (not bullet)
    text = re.sub(r'(?<!\n)\*([^*\n]+)\*', r'\1', text)
    # Clean up markdown table separators (|---|---|)
    text = re.sub(r'\|[-:]+\|[-:| ]+\|?\n?', '', text)
    # Convert table rows to readable format: | A | B | C | → A: B, C
    text = re.sub(r'^\|(.+)\|\s*$', lambda m: m.group(1).strip().replace(' | ', '  —  '), text, flags=re.MULTILINE)
    # Remove remaining lone | at start/end of lines
    text = re.sub(r'^\|\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\|$', '', text, flags=re.MULTILINE)
    # Remove horizontal rules (--- or ***)
    text = re.sub(r'^[\-\*]{3,}\s*$', '', text, flags=re.MULTILINE)
    # Clean up excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def format_response_for_telegram(result: dict, free_remaining: int | None = None) -> tuple[str, str | None]:
    """Format the LLM response for Telegram.

    Returns (main_text, footer_text). Main text is sent as plain text (LLM output
    may contain Markdown that breaks Telegram HTML parsing). Footer is sent
    separately as HTML with citations and free tier info.
    """
    if result["status"] == "error":
        return (f"Error procesando tu consulta:\n{result['response']}\n\nIntenta de nuevo o contacta /soporte", None)

    main_text = _clean_markdown(result["response"])

    # Append disclaimer to main text (plain text, always visible)
    main_text += (
        "\n\n---\n"
        "AVISO: Herramienta de apoyo clinico. "
        "NO reemplaza el criterio medico profesional. "
        "Verifique siempre contra guias oficiales."
    )

    # Build footer with citations and metadata
    footer_parts = []

    citations = result.get("citations", [])
    if citations:
        footer_parts.append("\n<b>Referencias:</b>")
        for i, cite in enumerate(citations[:5], 1):
            footer_parts.append(f"  {i}. <i>{_escape_html(cite)}</i>")

    elapsed = result.get("processing_time", 0)
    if elapsed:
        footer_parts.append(f"\n<i>{elapsed:.1f}s</i>")

    # Note about references in PDF (don't inline them in Telegram)
    if result.get("has_references"):
        footer_parts.append("\n<i>Se encontraron referencias cientificas. Descarga el PDF para verlas completas.</i>")

    if free_remaining is not None and free_remaining >= 0:
        if free_remaining > 0:
            footer_parts.append(f"\nConsultas gratis restantes: <b>{free_remaining}/5</b>")
        else:
            footer_parts.append(
                "\nHas usado tus 5 consultas gratis.\n"
                "Plan Basico $14.99 USD/mes | Plan Premium $24.99 USD/mes\n"
                "/suscribir para activar"
            )

    footer = "\n".join(footer_parts) if footer_parts else None
    return (main_text, footer)


def _escape_html(text: str) -> str:
    """Escape HTML special characters for Telegram."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_consultation_pdf(query_text: str, response_text: str,
                               citations: list[str], specialty: str,
                               processing_time: float = 0,
                               pubmed_papers: list[dict] = None) -> str:
    """Generate a PDF for a consultation. Returns path to the PDF file."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed, cannot generate PDF")
        return None

    pdf_dir = Path("data/bot_pdfs")
    pdf_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = str(pdf_dir / f"consulta_{timestamp}.pdf")

    doc = fitz.open()
    page_w, page_h = 612, 792  # Letter size
    margin_l, margin_r, margin_t = 55, 55, 50
    content_w = page_w - margin_l - margin_r

    page = doc.new_page(width=page_w, height=page_h)
    y = margin_t

    def new_page_if_needed(need_y):
        nonlocal page, y
        if y + need_y > page_h - 60:
            page = doc.new_page(width=page_w, height=page_h)
            y = margin_t
        return page

    # ── Header bar ──
    import fitz as fitz_mod
    header_rect = fitz_mod.Rect(0, 0, page_w, 38)
    page.draw_rect(header_rect, color=None, fill=(0.12, 0.25, 0.50))
    page.insert_text(fitz_mod.Point(margin_l, 26), "MedExpert",
                     fontsize=16, fontname="hebo", color=(1, 1, 1))
    page.insert_text(fitz_mod.Point(page_w - margin_r - 120, 26),
                     f"{specialty.capitalize()} | {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                     fontsize=8, fontname="helv", color=(0.8, 0.85, 0.95))
    y = 55

    # ── Consulta ──
    y = _pdf_section_title(page, "CONSULTA", margin_l, y, content_w)
    y = _pdf_body(doc, page, query_text, margin_l, y, content_w, fontsize=10)
    y += 12

    # ── Respuesta ──
    new_page_if_needed(30)
    y = _pdf_section_title(page, "RESPUESTA CLINICA", margin_l, y, content_w)

    # Clean markdown before PDF rendering
    clean_response = _clean_markdown(response_text)

    # Process response: detect section headers (MAYUSCULAS:) and bullet points
    for line in clean_response.split("\n"):
        new_page_if_needed(15)
        stripped = line.strip()
        if not stripped:
            y += 6
            continue
        # Section headers: lines that are ALL CAPS or end with ':'
        is_header = (stripped.isupper() and len(stripped) > 3) or (
            stripped.endswith(":") and stripped[:-1].replace(" ", "").isupper() and len(stripped) > 5)
        if is_header:
            y += 4
            y = _pdf_text(page, stripped, margin_l, y, content_w,
                          fontsize=10, fontname="hebo", color=(0.12, 0.25, 0.50))
            y += 2
        elif stripped.startswith(("- ", "• ", "* ")):
            # Bullet point — indent
            bullet_text = stripped[2:]
            y = _pdf_text(page, f"  \u2022  {bullet_text}", margin_l, y, content_w,
                          fontsize=9, color=(0.15, 0.15, 0.15))
        elif stripped.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.")):
            y = _pdf_text(page, f"  {stripped}", margin_l, y, content_w,
                          fontsize=9, color=(0.15, 0.15, 0.15))
        else:
            y = _pdf_text(page, stripped, margin_l, y, content_w,
                          fontsize=9, color=(0.1, 0.1, 0.1))

    # ── Referencias ──
    if citations:
        y += 10
        new_page_if_needed(30)
        y = _pdf_section_title(page, "REFERENCIAS", margin_l, y, content_w)
        for i, cite in enumerate(citations[:5], 1):
            new_page_if_needed(14)
            y = _pdf_text(page, f"  {i}. {cite}", margin_l, y, content_w,
                          fontsize=8, color=(0.35, 0.35, 0.35))

    # ── References (PubMed or Perplexity) ──
    if pubmed_papers:
        y += 10
        new_page_if_needed(30)
        # Check if we have a Perplexity summary
        perplexity_summary = pubmed_papers[0].get("_perplexity_summary", "") if pubmed_papers else ""
        if perplexity_summary:
            y = _pdf_section_title(page, "LITERATURA CIENTIFICA RECOMENDADA", margin_l, y, content_w)
            # Render Perplexity summary as body text
            for line in perplexity_summary.split("\n"):
                stripped = line.strip()
                if not stripped:
                    y += 4
                    continue
                new_page_if_needed(14)
                is_header = stripped.endswith(":") and len(stripped) < 80
                if is_header:
                    y = _pdf_text(page, stripped, margin_l, y, content_w,
                                  fontsize=8, fontname="hebo", color=(0.12, 0.25, 0.50))
                elif stripped.startswith(("- ", "• ", "* ")):
                    y = _pdf_text(page, f"  \u2022  {stripped[2:]}", margin_l, y, content_w,
                                  fontsize=8, color=(0.15, 0.15, 0.15))
                else:
                    y = _pdf_text(page, stripped, margin_l, y, content_w,
                                  fontsize=8, color=(0.1, 0.1, 0.1))
            # Add clickable reference links
            y += 8
            new_page_if_needed(20)
            y = _pdf_text(page, "ENLACES:", margin_l, y, content_w,
                          fontsize=8, fontname="hebo", color=(0.12, 0.25, 0.50))
            for i, paper in enumerate(pubmed_papers, 1):
                url = paper.get("doi_url") or paper.get("pubmed_url")
                if not url:
                    continue
                new_page_if_needed(14)
                link_text = f"  {i}. {paper['title'][:80]}{'...' if len(paper.get('title','')) > 80 else ''}"
                link_y = y
                y = _pdf_text(page, link_text, margin_l, y, content_w,
                              fontsize=7, color=(0.1, 0.4, 0.8))
                # Add clickable link annotation
                link_rect = fitz_mod.Rect(margin_l, link_y - 2, margin_l + content_w, y)
                page.insert_link({"kind": fitz_mod.LINK_URI, "from": link_rect, "uri": url})
                y += 2
        else:
            y = _pdf_section_title(page, "LITERATURA CIENTIFICA RECIENTE", margin_l, y, content_w)
            for i, paper in enumerate(pubmed_papers, 1):
                new_page_if_needed(60)
                # Title
                y = _pdf_text(page, f"  {i}. {paper['title']}", margin_l, y, content_w,
                              fontsize=8, fontname="hebo", color=(0.12, 0.25, 0.50))
                # Authors + year + journal
                meta_line = f"     {paper['authors']} ({paper['year']})"
                if paper.get("journal"):
                    meta_line += f" - {paper['journal']}"
                y = _pdf_text(page, meta_line, margin_l, y, content_w,
                              fontsize=7, color=(0.4, 0.4, 0.4))
                # Abstract (truncated)
                if paper.get("abstract"):
                    abstract = paper.get("abstract_es", paper["abstract"])
                    abstract_short = abstract[:300] + "..." if len(abstract) > 300 else abstract
                    new_page_if_needed(40)
                    y = _pdf_text(page, f"     {abstract_short}", margin_l, y, content_w,
                                  fontsize=7, color=(0.25, 0.25, 0.25))
                # DOI/PMID — clickable link
                url = paper.get("doi_url") or paper.get("pubmed_url")
                if url:
                    link_label = f"     {'DOI' if paper.get('doi_url') else 'PMID'}: {url}"
                    link_y = y
                    y = _pdf_text(page, link_label, margin_l, y, content_w,
                                  fontsize=7, color=(0.1, 0.4, 0.8))
                    link_rect = fitz_mod.Rect(margin_l, link_y - 2, margin_l + content_w, y)
                    page.insert_link({"kind": fitz_mod.LINK_URI, "from": link_rect, "uri": url})
                y += 4

    # ── Disclaimer footer on ALL pages ──
    disclaimer_text = "AVISO: Herramienta de apoyo clinico. NO reemplaza el criterio medico profesional."
    generated_text = f"Generado por MedExpert | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    for pg in doc:
        footer_y = page_h - 35
        pg.draw_line(fitz_mod.Point(margin_l, footer_y),
                     fitz_mod.Point(page_w - margin_r, footer_y),
                     color=(0.85, 0.85, 0.85), width=0.5)
        pg.insert_text(fitz_mod.Point(margin_l, footer_y + 12),
                       disclaimer_text,
                       fontsize=7, fontname="hebo", color=(0.4, 0.15, 0.15))
        pg.insert_text(fitz_mod.Point(margin_l, footer_y + 22),
                       generated_text,
                       fontsize=6, fontname="helv", color=(0.6, 0.6, 0.6))

    doc.save(pdf_path)
    doc.close()
    logger.info(f"PDF generated: {pdf_path}")
    return pdf_path


def _pdf_section_title(page, title: str, x: float, y: float, width: float) -> float:
    """Draw a section title with underline."""
    import fitz
    page.insert_text(fitz.Point(x, y + 11), title,
                     fontsize=11, fontname="hebo", color=(0.12, 0.25, 0.50))
    y += 15
    page.draw_line(fitz.Point(x, y), fitz.Point(x + width, y),
                   color=(0.12, 0.25, 0.50), width=0.8)
    y += 8
    return y


def _pdf_text(page, text: str, x: float, y: float, width: float,
              fontsize: int = 9, fontname: str = "helv",
              color: tuple = (0, 0, 0)) -> float:
    """Write a single text line with word wrapping. Returns new y."""
    import fitz
    if not text:
        return y + 4

    # Strip markdown bold markers for PDF
    clean = text.replace("**", "").replace("*", "")

    # Word wrap
    chars_per_line = int(width / (fontsize * 0.52))
    lines = _wrap_text(clean, chars_per_line)

    line_h = fontsize + 3
    for line in lines:
        if y + line_h > 750:
            return y
        page.insert_text(fitz.Point(x, y + fontsize), line,
                         fontsize=fontsize, fontname=fontname, color=color)
        y += line_h
    return y


def _pdf_body(doc, page, text: str, x: float, y: float, width: float,
              fontsize: int = 10) -> float:
    """Write body text, creating new pages as needed."""
    import fitz
    chars_per_line = int(width / (fontsize * 0.52))
    lines = _wrap_text(text, chars_per_line)
    line_h = fontsize + 3

    for line in lines:
        if y + line_h > 740:
            page = doc.new_page(width=612, height=792)
            y = 50
        page.insert_text(fitz.Point(x, y + fontsize), line,
                         fontsize=fontsize, fontname="helv", color=(0.15, 0.15, 0.15))
        y += line_h
    return y


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Word-wrap text into lines of max_chars width."""
    words = text.split()
    lines = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        if len(test) > max_chars:
            if current:
                lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


# ─────────────────────────────────────────────
# Bilingual RAG Search
# ─────────────────────────────────────────────

# Clinical term translation map (Spanish → English)
_CLINICAL_TRANSLATIONS = {
    # Cancer types (phrases — checked first as bigrams)
    "cáncer pulmonar": "lung cancer", "cancer pulmonar": "lung cancer",
    "cáncer de pulmón": "lung cancer", "cancer de pulmon": "lung cancer",
    "cáncer de mama": "breast cancer", "cancer de mama": "breast cancer",
    "cáncer de próstata": "prostate cancer", "cancer de prostata": "prostate cancer",
    "cáncer gástrico": "gastric cancer", "cancer gastrico": "gastric cancer",
    "cáncer de colon": "colon cancer", "cancer de colon": "colon cancer",
    "cáncer de ovario": "ovarian cancer", "cancer de ovario": "ovarian cancer",
    "cáncer de páncreas": "pancreatic cancer", "cancer de pancreas": "pancreatic cancer",
    "cáncer de hígado": "liver cancer", "cancer de higado": "liver cancer",
    "cáncer de riñón": "kidney cancer", "cancer de rinon": "kidney cancer",
    "cáncer de vejiga": "bladder cancer", "cancer de vejiga": "bladder cancer",
    "cáncer de tiroides": "thyroid cancer", "cancer de tiroides": "thyroid cancer",
    "cáncer cervicouterino": "cervical cancer", "cancer cervicouterino": "cervical cancer",
    "cabeza y cuello": "head and neck",
    # Cancer types (single words)
    "cáncer": "cancer", "cancer": "cancer", "carcinoma": "carcinoma",
    "tumor": "tumor", "tumores": "tumors", "neoplasia": "neoplasm",
    "linfoma": "lymphoma", "melanoma": "melanoma", "sarcoma": "sarcoma",
    "leucemia": "leukemia", "mieloma": "myeloma",
    # Organs
    "mama": "breast", "seno": "breast",
    "pulmón": "lung", "pulmon": "lung", "pulmonar": "lung",
    "próstata": "prostate", "prostata": "prostate",
    "colon": "colon", "recto": "rectum", "colorrectal": "colorectal",
    "hígado": "liver", "higado": "liver", "hepatocelular": "hepatocellular",
    "páncreas": "pancreas", "pancreas": "pancreas", "pancreático": "pancreatic",
    "riñón": "kidney", "rinon": "kidney", "renal": "renal",
    "vejiga": "bladder", "estómago": "stomach", "estomago": "stomach",
    "gástrico": "gastric", "gastrico": "gastric",
    "esófago": "esophagus", "esofago": "esophagus", "esofágico": "esophageal",
    "ovario": "ovary", "ovárico": "ovarian", "ovarico": "ovarian",
    "útero": "uterus", "utero": "uterus", "uterino": "uterine",
    "endometrio": "endometrium", "endometrial": "endometrial",
    "cérvix": "cervix", "cervix": "cervix", "cervical": "cervical",
    "testículo": "testicular", "testiculo": "testicular",
    "tiroides": "thyroid", "cabeza": "head", "cuello": "neck",
    "cerebro": "brain", "cerebral": "brain",
    # Treatments
    "tratamiento": "treatment", "quimioterapia": "chemotherapy",
    "radioterapia": "radiotherapy", "inmunoterapia": "immunotherapy",
    "cirugía": "surgery", "cirugia": "surgery", "quirúrgico": "surgical",
    "hormonoterapia": "hormone therapy", "terapia": "therapy",
    "adyuvante": "adjuvant", "neoadyuvante": "neoadjuvant",
    "paliativo": "palliative", "curativo": "curative",
    "primera línea": "first line", "primera linea": "first line",
    "segunda línea": "second line", "segunda linea": "second line",
    # Staging & diagnosis
    "estadio": "stage", "estadificación": "staging", "estadificacion": "staging",
    "metástasis": "metastasis", "metastasis": "metastasis",
    "metastásico": "metastatic", "metastasico": "metastatic",
    "localmente avanzado": "locally advanced",
    "diagnóstico": "diagnosis", "diagnostico": "diagnosis",
    "biopsia": "biopsy", "histología": "histology", "histologia": "histology",
    "diferenciado": "differentiated", "indiferenciado": "undifferentiated",
    "ganglios": "lymph nodes", "ganglio": "lymph node",
    "recurrente": "recurrent", "recurrencia": "recurrence",
    "supervivencia": "survival", "pronóstico": "prognosis", "pronostico": "prognosis",
    # Clinical terms
    "paciente": "patient", "masculino": "male", "femenino": "female",
    "años": "years", "edad": "age", "antígeno": "antigen", "antigeno": "antigen",
    "prostático": "prostatic", "prostatico": "prostatic",
    "específico": "specific", "especifico": "specific",
    "marcador": "marker", "marcadores": "markers",
    "elevado": "elevated", "nivel": "level", "niveles": "levels",
    # Adjective forms (critical for RAG matching)
    "agudo": "acute", "aguda": "acute",
    "crónico": "chronic", "cronico": "chronic", "crónica": "chronic", "cronica": "chronic",
    "linfoblástica": "lymphoblastic", "linfoblastica": "lymphoblastic",
    "linfoblástico": "lymphoblastic", "linfoblastico": "lymphoblastic",
    "mieloide": "myeloid", "mielógeno": "myelogenous", "mielogeno": "myelogenous",
    "folicular": "follicular", "difuso": "diffuse",
    "escamoso": "squamous", "escamosa": "squamous",
    "adenocarcinoma": "adenocarcinoma",
    "microcítico": "small cell", "microcitico": "small cell",
    "no microcítico": "non-small cell", "no microcitico": "non-small cell",
    "células pequeñas": "small cell", "celulas pequenas": "small cell",
    "células grandes": "large cell", "celulas grandes": "large cell",
    "negativo": "negative", "positivo": "positive",
    "triple negativo": "triple negative",
    "receptor": "receptor", "receptores": "receptors",
    "hormonal": "hormonal", "her2": "her2", "brca": "brca",
    "resecable": "resectable", "irresecable": "unresectable",
    "operable": "operable", "inoperable": "inoperable",
    "avanzado": "advanced", "avanzada": "advanced",
    "temprano": "early", "temprana": "early",
    "invasivo": "invasive", "invasiva": "invasive",
    "bilateral": "bilateral", "unilateral": "unilateral",
    "primario": "primary", "primaria": "primary",
    "secundario": "secondary", "secundaria": "secondary",
    # More clinical terms
    "dosis": "dose", "dosificación": "dosing", "dosificacion": "dosing",
    "ciclo": "cycle", "ciclos": "cycles",
    "respuesta": "response", "remisión": "remission", "remision": "remission",
    "completa": "complete", "parcial": "partial",
    "progresión": "progression", "progresion": "progression",
    "resistente": "resistant", "resistencia": "resistance",
    "toxicidad": "toxicity", "efectos": "effects", "adversos": "adverse",
    "neutropenia": "neutropenia", "febril": "febrile",
    "anemia": "anemia", "trombocitopenia": "thrombocytopenia",
    "neuropatía": "neuropathy", "neuropatia": "neuropathy",
    "náusea": "nausea", "nausea": "nausea", "vómito": "vomiting", "vomito": "vomiting",
    "dolor": "pain", "fatiga": "fatigue",
    # Guidelines
    "guía": "guideline", "guia": "guideline",
    "recomendación": "recommendation", "recomendacion": "recommendation",
    "manejo": "management", "seguimiento": "follow-up",
}


def _search_literature(query_text: str) -> list[dict]:
    """Search for clinical literature using configured provider (Perplexity or PubMed).

    Returns list of paper dicts compatible with PDF generation:
    {title, authors, year, abstract, abstract_es, doi_url, pubmed_url, journal, pmid}
    """
    settings = db.get_all_settings()
    perplexity_enabled = settings.get("search_perplexity_enabled") == "1"
    pubmed_enabled = settings.get("search_pubmed_enabled", "1") != "0"

    papers = []

    # Try Perplexity first if enabled
    if perplexity_enabled:
        papers = _search_perplexity(query_text, settings)
        if papers:
            return papers

    # Fall back to PubMed if enabled
    if pubmed_enabled:
        try:
            from pubmed import search_pubmed, translate_abstracts
            pubmed_query_en = _build_pubmed_query(query_text)
            if pubmed_query_en:
                logger.info(f"PubMed search: '{pubmed_query_en}'")
                papers = search_pubmed(pubmed_query_en, max_results=5)
                if papers:
                    papers = translate_abstracts(papers)
                    logger.info(f"PubMed: {len(papers)} papers found")
        except Exception as e:
            logger.warning(f"PubMed search failed (non-blocking): {e}")

    return papers


def _search_perplexity(query_text: str, settings: dict) -> list[dict]:
    """Search for clinical literature via Perplexity (Synapse).

    Returns papers in the same format as PubMed for PDF compatibility.
    """
    api_key = os.getenv("PERPLEXITY_API_KEY", "")
    base_url = os.getenv("PERPLEXITY_BASE_URL", "http://100.72.169.113:8800/v1")
    model = settings.get("search_perplexity_model_fast", "sonar")

    if not api_key:
        logger.warning("Perplexity enabled but no API key")
        return []

    try:
        from openai import OpenAI
        import json as _json

        client = OpenAI(base_url=base_url, api_key=api_key)

        # Truncate query
        query_short = query_text[:500] if len(query_text) > 500 else query_text

        prompt = (
            "Busca literatura cientifica publicada en JOURNALS MEDICOS PEER-REVIEWED "
            "sobre este caso clinico.\n\n"
            "REGLAS ESTRICTAS:\n"
            "- SOLO articulos de revistas medicas indexadas (PubMed, Scopus, Web of Science)\n"
            "- SOLO de los ultimos 5 años (2021-2026)\n"
            "- Priorizar: meta-analisis > revisiones sistematicas > ensayos clinicos (RCTs) > guias de practica clinica\n"
            "- Priorizar journals de alto impacto: NEJM, Lancet, Lancet Oncology, JCO, "
            "Annals of Oncology, Journal of Clinical Oncology, BMJ, JAMA, JAMA Oncology, "
            "European Journal of Cancer, Cancer Research, Nature Medicine\n"
            "- EXCLUIR: sitios web informativos, blogs medicos, panfletos, paginas de pacientes, "
            "noticias, Wikipedia, WebMD, Mayo Clinic web, Medscape articulos informativos\n"
            "- EXCLUIR: articulos anteriores a 2021\n\n"
            f"Caso clinico: {query_short}\n\n"
            "Responde con los 5 articulos mas relevantes y de mayor impacto. "
            "Para cada uno incluye:\n"
            "- Titulo completo del articulo\n"
            "- Autores principales (apellido et al.)\n"
            "- Revista (journal name)\n"
            "- Año de publicacion\n"
            "- Resumen breve (2-3 oraciones) en español de los hallazgos clave\n"
            "- DOI o enlace a PubMed"
        )

        response = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            timeout=60.0,
        )

        response_text = response.choices[0].message.content or ""
        model_used = getattr(response, "model", model) or model
        logger.info(f"Perplexity search OK ({model_used}), {len(response_text)} chars")

        # Extract citations from Perplexity response (returned as extra field)
        raw = response.model_dump() if hasattr(response, "model_dump") else {}
        citations_urls = raw.get("citations", [])
        search_results = raw.get("search_results", [])

        # Build papers list from search_results + citations
        papers = []
        if search_results:
            for sr in search_results[:8]:
                papers.append({
                    "title": sr.get("title", ""),
                    "authors": "",
                    "year": (sr.get("date", "") or sr.get("last_updated", ""))[:4],
                    "abstract": sr.get("snippet", ""),
                    "abstract_es": sr.get("snippet", ""),
                    "doi_url": sr.get("url", ""),
                    "pubmed_url": "",
                    "pmid": "",
                    "journal": sr.get("source", "web"),
                })
        elif citations_urls:
            # Fallback: use citation URLs
            for url in citations_urls[:8]:
                papers.append({
                    "title": url,
                    "authors": "",
                    "year": "",
                    "abstract": "",
                    "abstract_es": "",
                    "doi_url": url,
                    "pubmed_url": "",
                    "pmid": "",
                    "journal": "",
                })

        # Also store the full Perplexity response text for context
        if papers:
            papers[0]["_perplexity_summary"] = response_text

        logger.info(f"Perplexity: {len(papers)} references extracted")
        return papers

    except Exception as e:
        logger.warning(f"Perplexity search failed: {e}")
        return []


def _build_pubmed_query(original_query: str, followup_question: str = None) -> str:
    """Use Groq LLM to extract clinical keywords and build a PubMed search query."""
    try:
        from openai import OpenAI
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            return ""
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")

        clinical_text = original_query
        if followup_question:
            clinical_text += f"\nPregunta adicional: {followup_question}"

        # Truncate to avoid sending huge transcriptions
        if len(clinical_text) > 500:
            clinical_text = clinical_text[:500]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=30,
            messages=[
                {"role": "system", "content": (
                    "You extract clinical keywords from Spanish medical queries and return "
                    "ONLY a short PubMed search query in English (5-10 words). "
                    "Use standard medical/MeSH terms. No explanations, no quotes. "
                    "Examples:\n"
                    "Input: Paciente con cáncer de mama triple negativo estadio II\n"
                    "Output: triple negative breast cancer stage II treatment\n"
                    "Input: Carcinoma nasofaríngeo indiferenciado con ganglios cervicales\n"
                    "Output: undifferentiated nasopharyngeal carcinoma cervical lymph nodes"
                )},
                {"role": "user", "content": clinical_text},
            ],
            timeout=10.0,
        )
        query = response.choices[0].message.content.strip().strip('"\'')
        logger.info(f"PubMed query built by LLM: '{query}'")
        return query
    except Exception as e:
        logger.warning(f"PubMed query build failed: {e}")
        # Fallback to trigram translation
        import re
        clean = re.sub(r'\[(?:NOMBRE|DIRECCION|TELEFONO|EMAIL|CURP|RFC|NSS|TARJETA)\]', '', original_query)
        return _translate_query_to_english(clean.strip())


def _translate_query_to_english(text: str) -> str:
    """Translate a Spanish clinical query to English using term mapping.

    This is a fast, zero-latency alternative to LLM translation.
    Focuses on clinical terms that matter for RAG cosine similarity.
    """
    if not text:
        return ""

    words = text.lower().split()
    translated = []

    i = 0
    while i < len(words):
        # Try 3-word phrases first
        if i + 2 < len(words):
            trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
            if trigram in _CLINICAL_TRANSLATIONS:
                translated.append(_CLINICAL_TRANSLATIONS[trigram])
                i += 3
                continue

        # Try 2-word phrases
        if i + 1 < len(words):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram in _CLINICAL_TRANSLATIONS:
                translated.append(_CLINICAL_TRANSLATIONS[bigram])
                i += 2
                continue

        word = words[i]
        # Strip punctuation for lookup
        clean = word.strip(".,;:!?()[]")
        if clean in _CLINICAL_TRANSLATIONS:
            translated.append(_CLINICAL_TRANSLATIONS[clean])
        elif clean.isalpha() and len(clean) > 3:
            # Keep words that might be the same in both languages (medical terms)
            translated.append(clean)
        i += 1

    result = " ".join(translated)
    if result and result != text.lower():
        logger.info(f"Query translated: '{text[:60]}' -> '{result[:60]}'")
        return result
    return ""


# ─────────────────────────────────────────────
# RAG Source Diversity
# ─────────────────────────────────────────────

# Priority tiers: international guidelines first, then local
_SOCIETY_PRIORITY = {
    "NCCN": 1,
    "ESMO": 1,
    "NCI": 2,
    "ASCO": 2,
    "IMSS": 3,
    "CMCM": 2,
    "STGALLEN": 2,
}


def _detect_society(source: str) -> str:
    """Detect medical society from source name when metadata is missing.

    Many NCCN/NCI sources were indexed before society detection was added,
    so they have society='' or society='none' in ChromaDB metadata.
    """
    if not source:
        return ""
    s = source.upper()
    if "NCCN" in s:
        return "NCCN"
    if "ESMO" in s:
        return "ESMO"
    if "NCI" in s or "PDQ" in s:
        return "NCI"
    if "ASCO" in s:
        return "ASCO"
    if "IMSS" in s or "GPC" in s or "GER" in s:
        return "IMSS"
    if "CMCM" in s or "CONSENSO" in s and "MAMA" in s:
        return "CMCM"
    if "STGALLEN" in s or "ST GALLEN" in s or "ST. GALLEN" in s:
        return "STGALLEN"
    return ""


def _diversify_results(hits: list[dict], max_per_source: int = 3,
                       total: int = 10) -> list[dict]:
    """Select diverse RAG results prioritizing NCCN/ESMO over IMSS.

    Groups results by society, picks top results from each group
    following the priority order: NCCN/ESMO > NCI/ASCO > IMSS > other.
    Within each group, original ChromaDB ranking (cosine distance) is preserved.
    """
    if not hits:
        return []

    # Tag each hit with resolved society
    for hit in hits:
        if not hit.get("society") or hit["society"].lower() == "none":
            hit["society"] = _detect_society(hit.get("source", ""))

    # Group by society
    groups: dict[str, list[dict]] = {}
    for hit in hits:
        soc = hit.get("society", "") or "OTHER"
        groups.setdefault(soc, []).append(hit)

    # Build priority-ordered list of societies
    sorted_societies = sorted(
        groups.keys(),
        key=lambda s: (_SOCIETY_PRIORITY.get(s.upper(), 4), s),
    )

    # Round-robin pick from each society in priority order
    selected = []
    picked_per_society: dict[str, int] = {s: 0 for s in sorted_societies}

    # Pass 1: take up to max_per_source from each society
    for soc in sorted_societies:
        for hit in groups[soc]:
            if picked_per_society[soc] >= max_per_source:
                break
            if len(selected) >= total:
                break
            selected.append(hit)
            picked_per_society[soc] += 1
        if len(selected) >= total:
            break

    # Pass 2: if we still need more, fill from highest-priority groups
    if len(selected) < total:
        for soc in sorted_societies:
            for hit in groups[soc]:
                if hit in selected:
                    continue
                selected.append(hit)
                if len(selected) >= total:
                    break
            if len(selected) >= total:
                break

    return selected
