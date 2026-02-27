"""
MedExpert Telegram Bot - Main Entry Point
Telegram bot for clinical consultations via RAG + LLM.

Usage:
    python bot.py                    # Start bot (polling mode)
    python bot.py --specialty onco   # Specify specialty (default: oncologia)

Requires TELEGRAM_BOT_TOKEN in .env
"""

import os
import sys
import asyncio
import logging
import tempfile
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

import database as db
from bot_brain import BotBrain, transcribe_audio, format_response_for_telegram, generate_consultation_pdf

# Logging
logging.basicConfig(
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("medexpert.bot")
# Silence noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# Constants
FREE_QUERY_LIMIT = 5
PAID_DEEPEN_LIMIT_MONTHLY = 10  # Max deepenings per month for paid tier
DISCLAIMER_SHOWN_KEY = "disclaimer_shown"

# LLM brain (initialized on first query per specialty)
_brains: dict[str, BotBrain] = {}


def get_brain() -> BotBrain:
    """Get or create a BotBrain instance (re-reads settings each time for model changes)."""
    if "brain" not in _brains:
        _brains["brain"] = BotBrain()
    return _brains["brain"]


# ─────────────────────────────────────────────
# Telegram Handlers
# ─────────────────────────────────────────────

async def cmd_start(update, context):
    """Handle /start command — register user, show welcome."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    user = update.effective_user
    specialty = context.bot_data.get("specialty", "oncologia")

    # Register or retrieve user
    bot_user = db.get_or_create_bot_user(
        telegram_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        last_name=user.last_name or "",
        specialty=specialty,
    )

    # Check referral code in args
    if context.args:
        referral_code = context.args[0]
        if referral_code.startswith("ref_"):
            # TODO: Process referral in Sprint 2
            pass

    free_used = db.count_bot_free_queries(user.id, specialty)
    free_remaining = max(0, FREE_QUERY_LIMIT - free_used)

    # Get expert info for display
    expert = db.get_expert_by_slug(specialty)
    expert_name = expert["name"] if expert else specialty.capitalize()

    from rag_engine import get_rag_for_expert
    rag = get_rag_for_expert(specialty)
    guidelines_count = len(rag.list_guidelines())
    chunks_count = rag.get_total_count()

    welcome = (
        f"<b>MedExpert {expert_name}</b>\n\n"
        f"Hola Dr. {user.first_name}!\n\n"
        f"Soy tu asistente clinico especializado en {expert_name.lower()}.\n\n"
        f"<b>Tengo acceso a:</b>\n"
        f"  {guidelines_count} guias clinicas ({chunks_count:,} fragmentos indexados)\n"
        f"  Guidelines NCCN, ESMO, IMSS y mas\n\n"
        f"<b>Como usar:</b>\n"
        f"  Envia un audio con tu caso clinico\n"
        f"  O escribe directamente tu consulta\n\n"
        f"<b>IMPORTANTE:</b>\n"
        f"  Anonimiza datos del paciente.\n"
        f"  No nombres, telefonos ni direcciones.\n"
        f"  Solo datos clinicos relevantes.\n\n"
        f"Consultas gratis restantes: <b>{free_remaining}/{FREE_QUERY_LIMIT}</b>\n\n"
        f"/ayuda - Mas informacion\n"
        f"/estado - Estado de tu cuenta"
    )

    await update.message.reply_text(welcome, parse_mode="HTML")
    logger.info(f"User {user.id} (@{user.username}) started bot ({specialty})")


async def cmd_ayuda(update, context):
    """Handle /ayuda command."""
    help_text = (
        "<b>Ayuda - MedExpert Bot</b>\n\n"
        "<b>Como consultar:</b>\n"
        "  Envia un mensaje de audio o texto con tu caso clinico.\n"
        "  El bot buscara en guias clinicas y te dara una respuesta con referencias.\n\n"
        "<b>Tips para mejores resultados:</b>\n"
        "  Incluye edad, genero y comorbilidades\n"
        "  Menciona estadio o clasificacion si aplica\n"
        "  Se especifico en tu pregunta clinica\n\n"
        "<b>Privacidad:</b>\n"
        "  Tu audio se elimina tras transcribir\n"
        "  Las consultas se anonimizan\n"
        "  Datos eliminados automaticamente despues de 30 dias\n\n"
        "<b>Comandos:</b>\n"
        "  /start - Reiniciar bot\n"
        "  /ayuda - Esta ayuda\n"
        "  /estado - Estado de cuenta y consultas\n"
        "  /terminos - Terminos del servicio\n"
        "  /soporte - Contactar soporte\n\n"
        "<b>AVISO:</b> Herramienta de apoyo clinico.\n"
        "NO reemplaza el criterio medico profesional."
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def cmd_estado(update, context):
    """Handle /estado command — show account status."""
    user = update.effective_user
    specialty = context.bot_data.get("specialty", "oncologia")
    bot_user = db.get_bot_user(user.id)

    if not bot_user:
        await update.message.reply_text("No tienes cuenta. Envia /start primero.")
        return

    free_used = db.count_bot_free_queries(user.id, specialty)
    free_remaining = max(0, FREE_QUERY_LIMIT - free_used)
    verified_text = "Verificado" if bot_user.get("is_verified") else "No verificado"

    status = (
        f"<b>Estado de cuenta</b>\n\n"
        f"<b>Usuario:</b> {bot_user['first_name']} {bot_user.get('last_name', '')}\n"
        f"<b>Telegram:</b> @{bot_user.get('username', 'N/A')}\n"
        f"<b>Especialidad:</b> {specialty}\n"
        f"<b>Verificacion:</b> {verified_text}\n"
        f"<b>Codigo referido:</b> <code>{bot_user.get('referral_code', 'N/A')}</code>\n\n"
        f"<b>Consultas gratis:</b> {free_used}/{FREE_QUERY_LIMIT} usadas ({free_remaining} restantes)\n"
        f"<b>Suscripcion:</b> No activa\n\n"
        f"<b>Desde:</b> {bot_user.get('created_at', 'N/A')}\n"
        f"<b>Ultima actividad:</b> {bot_user.get('last_activity', 'N/A')}"
    )
    await update.message.reply_text(status, parse_mode="HTML")


async def cmd_terminos(update, context):
    """Handle /terminos command."""
    terms = (
        "<b>Terminos del Servicio - MedExpert</b>\n\n"
        "<b>1. Uso previsto</b>\n"
        "MedExpert es una herramienta de APOYO a la decision clinica. "
        "NO reemplaza el criterio medico profesional. "
        "El medico es siempre responsable de las decisiones clinicas.\n\n"
        "<b>2. Privacidad</b>\n"
        "  Audio eliminado inmediatamente tras transcribir\n"
        "  Consultas anonimizadas\n"
        "  Datos eliminados a los 30 dias\n"
        "  No compartimos datos con terceros\n\n"
        "<b>3. Responsabilidad del usuario</b>\n"
        "  Anonimizar datos de pacientes antes de enviar\n"
        "  No compartir credenciales de acceso\n"
        "  Verificar recomendaciones contra guias oficiales\n\n"
        "<b>4. Suscripcion</b>\n"
        "  5 consultas gratuitas de prueba\n"
        "  Plan Basico: $299 MXN/mes (consultas ilimitadas + profundizar con GPT-OSS 120B)\n"
        "  Plan Premium: $499 MXN/mes (todo + profundizar con Claude Opus 4.6)\n"
        "  Cancela cuando quieras, sin penalidad\n\n"
        "Contacto: /soporte"
    )
    await update.message.reply_text(terms, parse_mode="HTML")


async def cmd_soporte(update, context):
    """Handle /soporte command."""
    await update.message.reply_text(
        "<b>Soporte MedExpert</b>\n\n"
        "Para dudas, reportes o sugerencias:\n"
        "  Email: jmfraga@emergencias.com.mx\n\n"
        "Describe tu problema y te responderemos lo antes posible.",
        parse_mode="HTML",
    )


async def handle_voice(update, context):
    """Process voice message: download → transcribe → query → respond."""
    user = update.effective_user
    specialty = context.bot_data.get("specialty", "oncologia")

    # Check limits
    if not db.can_bot_user_query(user.id, specialty, FREE_QUERY_LIMIT):
        await _show_limit_reached(update)
        return

    # Ensure user exists
    db.get_or_create_bot_user(
        telegram_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        specialty=specialty,
    )

    # Show processing status
    processing_msg = await update.message.reply_text(
        "Transcribiendo audio...",
    )

    # Download voice file
    voice = update.message.voice or update.message.audio
    if not voice:
        await processing_msg.edit_text("No se detecto audio. Envia un mensaje de voz.")
        return

    try:
        file = await voice.get_file()
        # Save to temp file
        suffix = ".ogg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            await file.download_to_drive(tmp_path)

        # Transcribe
        transcript = await transcribe_audio(tmp_path, expert_slug=specialty)

        # Clean up audio immediately
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        if not transcript:
            await processing_msg.edit_text(
                "No se pudo transcribir el audio. Intenta de nuevo con un audio mas claro."
            )
            return

        duration = voice.duration or 0
        logger.info(f"Voice from {user.id}: {duration}s -> '{transcript[:80]}...'")

        # Update status
        await processing_msg.edit_text(
            f"Audio transcrito ({duration}s)\nBuscando en guias clinicas..."
        )

        # Query RAG + LLM
        brain = get_brain()
        result = brain.query(transcript, expert_slug=specialty)

        # Log consultation
        free_used = db.count_bot_free_queries(user.id, specialty)
        is_free = free_used < FREE_QUERY_LIMIT
        consultation_id = db.log_bot_consultation(
            telegram_id=user.id,
            specialty=specialty,
            query_type="voice",
            query_text=transcript,
            response_text=result.get("response", ""),
            response_time_seconds=result.get("processing_time", 0),
            llm_provider=result.get("provider", ""),
            llm_model=result.get("model", ""),
            tokens_input=result.get("token_usage", {}).get("input_tokens", 0),
            tokens_output=result.get("token_usage", {}).get("output_tokens", 0),
            rag_chunks_used=result.get("rag_chunks_used", 0),
            is_free_tier=is_free,
            citations=result.get("citations", []),
        )

        # Format and send response
        free_remaining = max(0, FREE_QUERY_LIMIT - free_used - 1)
        main_text, footer = format_response_for_telegram(
            result,
            free_remaining=free_remaining if is_free else None,
        )

        # Delete processing message and send response
        await processing_msg.delete()

        # Send main response as plain text (LLM output may have Markdown)
        await _send_long_message(update, main_text)

        # Send footer (citations, free tier) with action buttons
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [[
                InlineKeyboardButton(
                    "Profundizar", callback_data=f"deepen_{consultation_id}"
                ),
                InlineKeyboardButton(
                    "Exportar PDF", callback_data=f"pdf_{consultation_id}"
                ),
            ]]
            await update.message.reply_text(
                footer, parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

    except Exception as e:
        logger.error(f"Voice processing error: {e}", exc_info=True)
        try:
            await processing_msg.edit_text(
                "Error procesando audio. Intenta de nuevo.\n\nSi persiste: /soporte"
            )
        except Exception:
            pass


async def handle_text(update, context):
    """Process text message: query RAG + LLM → respond."""
    user = update.effective_user
    specialty = context.bot_data.get("specialty", "oncologia")
    text = update.message.text.strip()

    if not text:
        return

    # Check limits
    if not db.can_bot_user_query(user.id, specialty, FREE_QUERY_LIMIT):
        await _show_limit_reached(update)
        return

    # Ensure user exists
    db.get_or_create_bot_user(
        telegram_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        specialty=specialty,
    )

    # Show processing status
    processing_msg = await update.message.reply_text(
        "Buscando en guias clinicas..."
    )

    try:
        logger.info(f"Text from {user.id}: '{text[:80]}...'")

        # Query RAG + LLM
        brain = get_brain()
        result = brain.query(text, expert_slug=specialty)

        # Log consultation
        free_used = db.count_bot_free_queries(user.id, specialty)
        is_free = free_used < FREE_QUERY_LIMIT
        consultation_id = db.log_bot_consultation(
            telegram_id=user.id,
            specialty=specialty,
            query_type="text",
            query_text=text,
            response_text=result.get("response", ""),
            response_time_seconds=result.get("processing_time", 0),
            llm_provider=result.get("provider", ""),
            llm_model=result.get("model", ""),
            tokens_input=result.get("token_usage", {}).get("input_tokens", 0),
            tokens_output=result.get("token_usage", {}).get("output_tokens", 0),
            rag_chunks_used=result.get("rag_chunks_used", 0),
            is_free_tier=is_free,
            citations=result.get("citations", []),
        )

        # Format and send response
        free_remaining = max(0, FREE_QUERY_LIMIT - free_used - 1)
        main_text, footer = format_response_for_telegram(
            result,
            free_remaining=free_remaining if is_free else None,
        )

        # Delete processing message and send response
        await processing_msg.delete()

        # Send main response as plain text (LLM output may have Markdown)
        await _send_long_message(update, main_text)

        # Send footer (citations, free tier) with action buttons
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [[
                InlineKeyboardButton(
                    "Profundizar", callback_data=f"deepen_{consultation_id}"
                ),
                InlineKeyboardButton(
                    "Exportar PDF", callback_data=f"pdf_{consultation_id}"
                ),
            ]]
            await update.message.reply_text(
                footer, parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

    except Exception as e:
        logger.error(f"Text processing error: {e}", exc_info=True)
        try:
            await processing_msg.edit_text(
                "Error procesando consulta. Intenta de nuevo.\n\nSi persiste: /soporte"
            )
        except Exception:
            pass


async def _show_limit_reached(update):
    """Show message when free queries are exhausted."""
    msg = (
        "<b>Limite de consultas gratuitas alcanzado</b>\n\n"
        "Has usado tus 5 consultas gratis.\n\n"
        "<b>Plan Basico - $299 MXN/mes</b>\n"
        "  Consultas ilimitadas\n"
        "  Profundizar 1 vez al dia con GPT-OSS 120B\n"
        "  Respuesta en menos de 5 segundos\n"
        "  Cancela cuando quieras\n\n"
        "<b>Plan Premium - $499 MXN/mes</b>\n"
        "  Todo lo del Plan Basico\n"
        "  Profundizar 1 vez al dia con Claude Opus 4.6\n"
        "  Respuestas de maxima calidad clinica\n"
        "  Soporte prioritario\n\n"
        "/suscribir para activar\n\n"
        "Invita colegas con tu codigo de referido: /estado"
    )
    await update.message.reply_text(msg, parse_mode="HTML")


async def _send_long_message(update, text: str, max_len: int = 4000):
    """Send a message as plain text, splitting if it exceeds Telegram's limit."""
    if len(text) <= max_len:
        await update.message.reply_text(text)
        return

    # Split on double newlines or single newlines
    parts = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                parts.append(current)
            current = line
        else:
            current += ("\n" + line) if current else line
    if current:
        parts.append(current)

    for part in parts:
        await update.message.reply_text(part)


async def handle_deepen_callback(update, context):
    """Handle 'Profundizar' button — tiered deepening.

    Free + basic plan: GPT-OSS 120B via Groq (~3s)
    Premium plan: Claude Opus 4.6 (~30s), max 1/day
    """
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("deepen_"):
        return

    consultation_id = int(data.split("_")[1])
    consultation = db.get_bot_consultation_by_id(consultation_id)

    if not consultation:
        await query.edit_message_text("Consulta no encontrada.")
        return

    user_id = query.from_user.id
    if consultation["telegram_id"] != user_id:
        await query.edit_message_text("No tienes acceso a esta consulta.")
        return

    specialty = consultation.get("specialty", "oncologia")

    # Check if already deepened
    if consultation.get("is_deepening"):
        await query.message.reply_text("Esta consulta ya fue profundizada.")
        return

    # Determine user tier and deepening model
    # TODO: Check actual subscription status when billing is implemented
    bot_user = db.get_bot_user(user_id)
    user_plan = "free"  # Will be: "free", "basic", "premium"

    if user_plan == "premium":
        # Premium: Opus, max 1/day
        opus_today = db.count_bot_opus_deepenings_today(user_id, specialty)
        if opus_today >= 1:
            await query.message.reply_text(
                "Ya usaste tu profundizacion premium (Opus) de hoy.\n"
                "Se renueva manana. Puedes seguir consultando normalmente."
            )
            return
        tier = "premium"
    else:
        # Free / Basic: GPT-OSS 120B
        tier = "free"

    # Free tier: deepening counts as a query
    free_used = db.count_bot_free_queries(user_id, specialty)
    is_free = user_plan == "free"

    if is_free:
        if not db.can_bot_user_query(user_id, specialty, FREE_QUERY_LIMIT):
            await query.message.reply_text(
                "No tienes consultas gratis restantes.\n"
                "Profundizar consume 1 consulta.\n\n"
                "Plan Basico ($299/mes): consultas ilimitadas + profundizar con GPT-OSS 120B\n"
                "Plan Premium ($499/mes): todo + profundizar con Claude Opus 4.6\n\n"
                "/suscribir para activar"
            )
            return

    # Show processing message
    model_label = "Claude Opus 4.6" if tier == "premium" else "GPT-OSS 120B"
    processing_msg = await query.message.reply_text(
        f"Profundizando con {model_label}..."
    )

    try:
        brain = get_brain()
        result = brain.deepen(
            original_query=consultation["query_text"],
            original_response=consultation["response_text"],
            expert_slug=specialty,
            tier=tier,
        )

        # Log as consultation (counts against free tier)
        deepen_id = db.log_bot_consultation(
            telegram_id=user_id,
            specialty=specialty,
            query_type=consultation.get("query_type", "text"),
            query_text=consultation["query_text"],
            response_text=result.get("response", ""),
            response_time_seconds=result.get("processing_time", 0),
            llm_provider=result.get("provider", ""),
            llm_model=result.get("model", ""),
            tokens_input=result.get("token_usage", {}).get("input_tokens", 0),
            tokens_output=result.get("token_usage", {}).get("output_tokens", 0),
            rag_chunks_used=result.get("rag_chunks_used", 0),
            is_free_tier=is_free,
            citations=result.get("citations", []),
            is_deepening=True,
            parent_consultation_id=consultation_id,
        )

        # Format and send
        main_text, footer = format_response_for_telegram(result)

        await processing_msg.delete()

        # Send deepened response
        await _send_long_message_from_callback(query, main_text)

        # Send footer with PDF button (no deepen button on deepened response)
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [[InlineKeyboardButton(
                "Exportar PDF", callback_data=f"pdf_{deepen_id}"
            )]]
            await query.message.reply_text(
                footer, parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )

        # Update original message to show it was deepened
        try:
            await query.edit_message_text(
                query.message.text_html or query.message.text or "Profundizado",
                parse_mode="HTML",
            )
        except Exception:
            pass

        logger.info(f"Deepen for {user_id} (tier={tier}, model={model_label})")

    except Exception as e:
        logger.error(f"Deepen error: {e}", exc_info=True)
        try:
            await processing_msg.edit_text(
                "Error al profundizar. Intenta de nuevo.\n\nSi persiste: /soporte"
            )
        except Exception:
            pass


async def _send_long_message_from_callback(callback_query, text: str, max_len: int = 4000):
    """Send a long message from a callback query context."""
    if len(text) <= max_len:
        await callback_query.message.reply_text(text)
        return

    parts = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            if current:
                parts.append(current)
            current = line
        else:
            current += ("\n" + line) if current else line
    if current:
        parts.append(current)

    for part in parts:
        await callback_query.message.reply_text(part)


async def handle_pdf_callback(update, context):
    """Handle PDF export button callback."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("pdf_"):
        return

    consultation_id = int(data.split("_")[1])
    consultation = db.get_bot_consultation_by_id(consultation_id)

    if not consultation:
        await query.edit_message_text("Consulta no encontrada.")
        return

    # Verify user owns this consultation
    if consultation["telegram_id"] != query.from_user.id:
        await query.edit_message_text("No tienes acceso a esta consulta.")
        return

    specialty = consultation.get("specialty", "oncologia")
    import json
    try:
        citations = json.loads(consultation.get("citations_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        citations = []
    pdf_path = generate_consultation_pdf(
        query_text=consultation["query_text"],
        response_text=consultation["response_text"],
        citations=citations,
        specialty=specialty,
        processing_time=consultation.get("response_time_seconds", 0),
    )

    if not pdf_path:
        await query.edit_message_text("Error generando PDF. PyMuPDF no disponible.")
        return

    try:
        await query.message.reply_document(
            document=open(pdf_path, "rb"),
            filename=f"MedExpert_{specialty}_{consultation_id}.pdf",
            caption=f"Consulta #{consultation_id} - MedExpert {specialty.capitalize()}",
        )
    except Exception as e:
        logger.error(f"PDF send error: {e}")
        await query.message.reply_text("Error enviando PDF.")
    finally:
        # Clean up PDF file
        try:
            os.unlink(pdf_path)
        except OSError:
            pass


async def handle_error(update, context):
    """Handle errors in the bot."""
    logger.error(f"Bot error: {context.error}", exc_info=context.error)
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Ocurrio un error inesperado. Intenta de nuevo.\n"
                "Si persiste: /soporte"
            )
        except Exception:
            pass


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    from telegram.ext import Application, CommandHandler, MessageHandler, filters

    parser = argparse.ArgumentParser(description="MedExpert Telegram Bot")
    parser.add_argument("--specialty", default=None, help="Expert specialty slug (default: oncologia)")
    args = parser.parse_args()

    specialty = args.specialty or os.getenv("BOT_SPECIALTY", "oncologia")
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)

    # Initialize database
    db.init_db()

    # Verify expert exists
    expert = db.get_expert_by_slug(specialty)
    if expert:
        logger.info(f"Bot specialty: {expert['name']} ({specialty})")
    else:
        logger.warning(f"Expert '{specialty}' not found in database. Bot will use default prompts.")

    # Build application
    app = Application.builder().token(token).build()
    app.bot_data["specialty"] = specialty

    # Register handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("ayuda", cmd_ayuda))
    app.add_handler(CommandHandler("help", cmd_ayuda))
    app.add_handler(CommandHandler("estado", cmd_estado))
    app.add_handler(CommandHandler("terminos", cmd_terminos))
    app.add_handler(CommandHandler("soporte", cmd_soporte))

    # Message handlers (voice and text)
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Callback handlers (PDF export button)
    from telegram.ext import CallbackQueryHandler
    app.add_handler(CallbackQueryHandler(handle_deepen_callback, pattern=r"^deepen_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_pdf_callback, pattern=r"^pdf_\d+$"))

    # Error handler
    app.add_error_handler(handle_error)

    # Start
    logger.info(f"MedExpert Bot starting (specialty: {specialty}, polling mode)")
    logger.info("Send /start to the bot on Telegram to begin")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
