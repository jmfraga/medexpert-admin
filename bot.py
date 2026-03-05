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

import json as _json
import database as db
from bot_brain import BotBrain, transcribe_audio, format_response_for_telegram, generate_consultation_pdf
from anonymizer import anonymize_text
from clinical_metadata import extract as extract_clinical_metadata

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

# Source preference constants
_ALL_SOURCES = ["NCCN", "ESMO", "NCI", "IMSS"]
_SOURCE_LABELS = {
    "NCCN": "NCCN (Natl. Comprehensive Cancer Network)",
    "ESMO": "ESMO (European Society Medical Oncology)",
    "NCI": "NCI/PDQ (National Cancer Institute)",
    "IMSS": "IMSS/GPC (Guias Practica Clinica)",
}

# LLM brain (initialized on first query per specialty)
_brains: dict[str, BotBrain] = {}


def get_brain(expert_slug: str = "oncologia") -> BotBrain:
    """Get or create a BotBrain instance per expert. Re-creates if config changed."""
    config = db.get_expert_llm_config(expert_slug)
    cache_key = f"{expert_slug}:{config['base_provider']}:{config['base_model']}:{config['deepen_provider']}:{config['deepen_model']}"
    if cache_key not in _brains:
        # Clear old entries for this expert
        _brains.clear()
        _brains[cache_key] = BotBrain(
            provider=config["base_provider"],
            model=config["base_model"],
            deepen_provider=config["deepen_provider"],
            deepen_model=config["deepen_model"],
        )
    return _brains[cache_key]


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

    # Check args (referral, payment redirect)
    if context.args:
        arg = context.args[0]
        if arg == "payment_success":
            context.user_data.pop("promo", None)
            await update.message.reply_text(
                "Pago recibido! Tu suscripcion se activara en unos momentos.\n"
                "Usa /estado para verificar."
            )
            return
        elif arg == "payment_cancel":
            await update.message.reply_text(
                "Pago cancelado. Puedes intentar de nuevo con /suscribir."
            )
            return
        elif arg.startswith("ref_"):
            ref_code = arg[4:]  # Strip "ref_" prefix
            result = db.process_referral(ref_code, user.id)
            if result:
                await update.message.reply_text(
                    "¡Fuiste referido! Ambos recibirán beneficios cuando te suscribas."
                )
                logger.info(f"Referral processed: {ref_code} -> {user.id}")
            # Continue to show welcome

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
        f"¡Hola Dr. {user.first_name}!\n\n"
        f"Soy tu asistente clínico especializado en {expert_name.lower()}, "
        f"respaldado por {guidelines_count} guías clínicas internacionales "
        f"(NCCN, ESMO, IMSS y más).\n\n"

        f"<b>¿Cómo consultar?</b>\n"
        f"  Envía un <b>audio de voz</b> o un <b>mensaje de texto</b> "
        f"con tu caso clínico. Recibirás una respuesta basada en "
        f"evidencia con referencias.\n\n"

        f"<b>Recomendaciones para mejores resultados:</b>\n"
        f"  Anonimiza siempre los datos del paciente\n"
        f"  Si usas audio, habla claro y pausado\n"
        f"  Usa nombres genéricos de medicamentos de preferencia\n"
        f"  Incluye edad, género, estadio y comorbilidades\n"
        f"  Sé específico en tu pregunta clínica\n\n"

        f"<b>Después de cada respuesta puedes:</b>\n"
        f"  <b>Profundizar</b> — análisis más detallado con IA avanzada\n"
        f"  <b>Exportar PDF</b> — documento con citas para tu expediente\n"
        f"  <b>Evaluar</b> — tu feedback nos ayuda a mejorar\n\n"

        f"<b>Comandos disponibles:</b>\n"
        f"  /ayuda — Guía completa y tips\n"
        f"  /estado — Tu cuenta y consultas restantes\n"
        f"  /suscribir — Planes y precios\n"
        f"  /congresos — Proximos congresos medicos\n"
        f"  /soporte — Reportar problemas o sugerencias\n"
        f"  /terminos — Aviso legal\n"
        f"  /cancelar — Cancelar acción en curso\n\n"

        f"Consultas gratis restantes: <b>{free_remaining}/{FREE_QUERY_LIMIT}</b>\n\n"

        f"<i>Herramienta de apoyo clínico. No sustituye el criterio "
        f"médico profesional.</i>"
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
        "  /suscribir - Planes y suscripciones\n"
        "  /congresos - Proximos congresos medicos\n"
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
        f"<b>Email:</b> {bot_user.get('email') or 'No registrado'}\n"
        f"<b>Especialidad:</b> {specialty}\n"
        f"<b>Verificacion:</b> {verified_text}\n"
        f"<b>Codigo referido:</b> <code>{bot_user.get('referral_code', 'N/A')}</code>\n\n"
        f"<b>Consultas gratis:</b> {free_used}/{FREE_QUERY_LIMIT} usadas ({free_remaining} restantes)\n"
    )
    # Show subscription status
    user_plan = db.get_bot_user_plan(user.id)
    if user_plan in ("basic", "premium"):
        expires = bot_user.get("subscription_expires_at", "N/A")
        status += f"<b>Suscripcion:</b> Plan {user_plan.capitalize()} (activa)\n"
        status += f"<b>Vence:</b> {expires}\n\n"
    else:
        status += f"<b>Suscripcion:</b> No activa (/suscribir)\n\n"
    status += (
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
        "  Plan Basico: $14.99 USD/mes (consultas ilimitadas + profundizar con GPT-OSS 120B)\n"
        "  Plan Premium: $24.99 USD/mes (todo + profundizar con Claude Opus 4.6)\n"
        "  Cancela cuando quieras, sin penalidad\n\n"
        "Contacto: /soporte"
    )
    await update.message.reply_text(terms, parse_mode="HTML")


async def cmd_soporte(update, context):
    """Handle /soporte command — ask user to describe their issue."""
    context.user_data["awaiting_support"] = True
    await update.message.reply_text(
        "<b>Soporte MedExpert</b>\n\n"
        "Describe tu problema, duda o sugerencia en un mensaje y lo enviaremos a nuestro equipo.\n\n"
        "Escribe /cancelar para cancelar.",
        parse_mode="HTML",
    )


async def cmd_cancelar(update, context):
    """Handle /cancelar — cancel any pending action."""
    cancelled = False
    for key in ("awaiting_support", "awaiting_email", "awaiting_verification", "promo", "pending_deepen"):
        if context.user_data.pop(key, None):
            cancelled = True
    if cancelled:
        await update.message.reply_text("Accion cancelada.")
    else:
        await update.message.reply_text("No hay ninguna accion pendiente.")


async def cmd_verificar(update, context):
    """Handle /verificar — start medical verification flow."""
    user = update.effective_user
    bot_user = db.get_bot_user(user.id)

    if bot_user and bot_user.get("is_verified"):
        await update.message.reply_text("Ya estás verificado como profesional de la salud.")
        return

    if bot_user and bot_user.get("verification_status") == "pending":
        await update.message.reply_text(
            "Tu verificación está en revisión.\n"
            "Te notificaremos cuando sea aprobada."
        )
        return

    context.user_data["awaiting_verification"] = "cedula"
    await update.message.reply_text(
        "<b>Verificación Médica</b>\n\n"
        "Para verificar tu identidad como profesional de la salud, necesitamos:\n\n"
        "1️⃣ <b>Cédula profesional</b> — foto legible\n"
        "2️⃣ <b>INE/Identificación oficial</b> — foto legible\n\n"
        "📷 Envía ahora la <b>foto de tu cédula profesional</b>.\n\n"
        "<i>Tus documentos se almacenan de forma segura y solo serán revisados "
        "por el equipo de MedExpert.</i>\n\n"
        "Usa /cancelar para cancelar.",
        parse_mode="HTML",
    )


async def handle_verification_photo(update, context):
    """Handle photo upload during verification flow."""
    step = context.user_data.get("awaiting_verification")
    if not step:
        return False  # Not in verification flow

    user = update.effective_user
    photo = update.message.photo[-1] if update.message.photo else None
    document = update.message.document if not photo else None

    if not photo and not document:
        await update.message.reply_text("Por favor envía una foto o documento.")
        return True

    # Download file
    if photo:
        file = await context.bot.get_file(photo.file_id)
    else:
        file = await context.bot.get_file(document.file_id)

    # Save to verification directory
    verify_dir = Path(f"data/verifications/{user.id}")
    verify_dir.mkdir(parents=True, exist_ok=True)
    ext = ".jpg" if photo else (Path(document.file_name).suffix if document.file_name else ".pdf")
    filepath = verify_dir / f"{step}{ext}"
    await file.download_to_drive(str(filepath))

    # Save to DB
    db.create_verification_doc(user.id, step, str(filepath))

    if step == "cedula":
        context.user_data["awaiting_verification"] = "ine"
        await update.message.reply_text(
            "✅ Cédula recibida.\n\n"
            "📷 Ahora envía la <b>foto de tu INE</b> (identificación oficial).",
            parse_mode="HTML",
        )
    elif step == "ine":
        context.user_data.pop("awaiting_verification", None)
        # Update user status
        db.update_bot_user(user.id, verification_status="pending")
        await update.message.reply_text(
            "✅ INE recibida.\n\n"
            "Tus documentos están en revisión. Te notificaremos cuando sean aprobados.\n"
            "Esto generalmente toma menos de 24 horas."
        )
        logger.info(f"Verification docs submitted by {user.id}")
    return True


async def cmd_congresos(update, context):
    """Handle /congresos command — show upcoming medical congresses."""
    events = db.get_upcoming_congresses(days_ahead=180)
    if not events:
        await update.message.reply_text(
            "No hay congresos programados en los proximos meses.\n"
            "Te notificaremos cuando se agreguen nuevos eventos."
        )
        return

    lines = ["<b>Congresos proximos</b>\n"]
    for e in events:
        lines.append(f"<b>{e['short_name'] or e['name']}</b>")
        lines.append(f"  {e['start_date']}")
        if e.get("end_date"):
            lines[-1] += f" - {e['end_date']}"
        if e.get("location"):
            lines.append(f"  {e['location']}")
        if e.get("url"):
            lines.append(f"  {e['url']}")
        lines.append("")

    lines.append("<i>Suscribete para recibir alertas y resumenes post-congreso.</i>")
    await update.message.reply_text("\n".join(lines), parse_mode="HTML")


async def cmd_suscribir(update, context):
    """Handle /suscribir command — show subscription plans with Stripe checkout."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    user = update.effective_user
    current_plan = db.get_bot_user_plan(user.id)

    if current_plan in ("basic", "premium"):
        await update.message.reply_text(
            f"Ya tienes una suscripcion activa: <b>Plan {current_plan.capitalize()}</b>\n\n"
            "Para cancelar o cambiar de plan: /soporte",
            parse_mode="HTML",
        )
        return

    # Check if we have their email
    bot_user = db.get_bot_user(user.id)
    if not bot_user or not bot_user.get("email"):
        context.user_data["awaiting_email"] = True
        await update.message.reply_text(
            "Antes de suscribirte, necesito tu email.\n"
            "Sera usado para tu facturacion y notificaciones.\n\n"
            "Escribe tu email:"
        )
        return

    promo = context.user_data.get("promo")
    prices = get_plan_prices()

    keyboard = []
    for key in ("basic_monthly", "basic_annual", "premium_monthly", "premium_annual"):
        info = prices[key]
        d_usd, _, pid = apply_promo_discount(promo, key, info["usd"], info["mxn"])
        if pid:
            label = f"{info['label']} - ${d_usd:.2f} USD (antes ${info['usd']:.2f})"
        else:
            label = f"{info['label']} - ${info['usd']:.2f} USD"
            if "annual" in key:
                label += " (2 meses gratis)"
        keyboard.append([InlineKeyboardButton(label, callback_data=f"subscribe_{key}")])

    promo_banner = ""
    if promo:
        if promo.get("discount_percent", 0) > 0:
            promo_banner = f"\nCodigo <b>{promo['code']}</b> aplicado: {promo['discount_percent']}% descuento\n"
        elif promo.get("discount_amount_usd", 0) > 0:
            promo_banner = f"\nCodigo <b>{promo['code']}</b> aplicado: ${promo['discount_amount_usd']:.2f} USD descuento\n"
    hint = "" if promo else "\n<i>¿Tienes un codigo? Usa /codigo CODIGO antes de elegir plan</i>\n"

    await update.message.reply_text(
        "<b>Suscripciones MedExpert</b>\n"
        f"{promo_banner}{hint}\n"
        "<b>Plan Basico</b> (precio de lanzamiento)\n"
        "  Consultas ilimitadas\n"
        "  Profundizar con GPT-OSS 120B\n"
        "  Exportar a PDF\n"
        "  Cancela cuando quieras\n\n"
        "<b>Plan Premium</b> (precio de lanzamiento)\n"
        "  Todo lo del Plan Basico\n"
        "  Profundizar con Claude Opus 4.6\n"
        "  Respuestas de maxima calidad clinica\n"
        "  Soporte prioritario\n\n"
        "Selecciona plan y periodo:",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


def get_plan_prices():
    """Load prices from DB, fallback to defaults."""
    try:
        prices = db.get_plan_prices_for_bot()
        if prices:
            return prices
    except Exception:
        pass
    return {
        "basic_monthly": {"usd": 14.99, "mxn": 299, "label": "Basico Mensual", "period": "mes"},
        "basic_annual": {"usd": 149.90, "mxn": 2990, "label": "Basico Anual", "period": "año"},
        "premium_monthly": {"usd": 24.99, "mxn": 499, "label": "Premium Mensual", "period": "mes"},
        "premium_annual": {"usd": 249.90, "mxn": 4990, "label": "Premium Anual", "period": "año"},
    }

PLAN_PRICES = get_plan_prices()


def apply_promo_discount(promo, plan_key, usd, mxn):
    """Apply promo discount to prices.
    Returns (discounted_usd, discounted_mxn, promo_id) or originals with None."""
    if not promo:
        return usd, mxn, None
    applies_to = promo.get("applies_to", "all")
    if applies_to != "all":
        plan_base = plan_key.split("_")[0]  # "basic" or "premium"
        if plan_base not in applies_to:
            return usd, mxn, None
    promo_id = promo["id"]
    if promo.get("discount_percent", 0) > 0:
        factor = 1 - (promo["discount_percent"] / 100)
        return round(usd * factor, 2), round(mxn * factor, 0), promo_id
    elif promo.get("discount_amount_usd", 0) > 0:
        discount_usd = promo["discount_amount_usd"]
        discount_mxn = discount_usd * 20  # Rough USD→MXN rate
        return max(0.01, round(usd - discount_usd, 2)), max(1, round(mxn - discount_mxn, 0)), promo_id
    return usd, mxn, None


async def cmd_codigo(update, context):
    """Handle /codigo CODE — apply a promo code before subscribing."""
    user = update.effective_user

    if not context.args:
        await update.message.reply_text(
            "Uso: /codigo CODIGO\n"
            "Ejemplo: /codigo TEST20"
        )
        return

    current_plan = db.get_bot_user_plan(user.id)
    if current_plan in ("basic", "premium"):
        await update.message.reply_text(
            "Ya tienes una suscripcion activa. Los codigos son solo para nuevas suscripciones."
        )
        return

    code = context.args[0].strip().upper()
    promo = db.validate_promo_code(code)

    if not promo:
        await update.message.reply_text(
            "Codigo no valido o expirado.\n"
            "Verifica e intenta de nuevo."
        )
        return

    context.user_data["promo"] = promo

    if promo.get("discount_percent", 0) > 0:
        desc = f"{promo['discount_percent']}% de descuento"
    elif promo.get("discount_amount_usd", 0) > 0:
        desc = f"${promo['discount_amount_usd']:.2f} USD de descuento"
    else:
        desc = promo.get("description", "Descuento aplicado")

    scope = ""
    applies = promo.get("applies_to", "all")
    if applies != "all":
        scope = f"\nAplica a: Plan {applies.capitalize()}"

    await update.message.reply_text(
        f"Codigo <b>{code}</b> aplicado: {desc}{scope}\n\n"
        "Ahora usa /suscribir para ver los precios con descuento.",
        parse_mode="HTML",
    )


async def handle_subscribe_callback(update, context):
    """Handle subscription plan selection — ask country for smart payment routing."""
    global PLAN_PRICES
    PLAN_PRICES = get_plan_prices()  # Reload from DB

    query = update.callback_query
    await query.answer()

    # e.g. "subscribe_basic_monthly"
    plan_key = query.data.replace("subscribe_", "")
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    promo = context.user_data.get("promo")
    d_usd, d_mxn, pid = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    mx_label = f"Mexico - ${d_mxn:.0f} MXN" if pid else f"Mexico - ${price_info['mxn']:.0f} MXN"
    intl_label = f"Internacional - ${d_usd:.2f} USD" if pid else f"Internacional - ${price_info['usd']:.2f} USD"
    keyboard = [
        [InlineKeyboardButton(mx_label, callback_data=f"region_mx_{plan_key}")],
        [InlineKeyboardButton(intl_label, callback_data=f"region_intl_{plan_key}")],
    ]

    await query.message.reply_text(
        f"<b>{price_info['label']}</b>\n\n"
        "¿Desde donde nos contactas?",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_region_mx(update, context):
    """Show Mexican payment methods: Mercado Pago + Clip."""
    query = update.callback_query
    await query.answer()

    plan_key = query.data.replace("region_mx_", "")
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    promo = context.user_data.get("promo")
    _, d_mxn, pid = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])
    show_mxn = d_mxn if pid else price_info["mxn"]

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton(
            f"Mercado Pago - ${show_mxn:.0f} MXN",
            callback_data=f"pay_mp_{plan_key}",
        )],
        [InlineKeyboardButton(
            f"Clip (OXXO/Tarjeta) - ${show_mxn:.0f} MXN",
            callback_data=f"pay_clip_{plan_key}",
        )],
    ]

    await query.message.reply_text(
        f"<b>{price_info['label']} - ${show_mxn:.0f} MXN/{price_info['period']}</b>\n\n"
        "Selecciona tu metodo de pago:",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_region_intl(update, context):
    """Show international payment methods: Stripe + PayPal."""
    query = update.callback_query
    await query.answer()

    plan_key = query.data.replace("region_intl_", "")
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    promo = context.user_data.get("promo")
    d_usd, _, pid = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])
    show_usd = d_usd if pid else price_info["usd"]

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton(
            f"Tarjeta (Stripe) - ${show_usd:.2f} USD",
            callback_data=f"pay_stripe_{plan_key}",
        )],
        [InlineKeyboardButton(
            f"PayPal - ${show_usd:.2f} USD",
            callback_data=f"pay_paypal_{plan_key}",
        )],
    ]

    await query.message.reply_text(
        f"<b>{price_info['label']} - ${show_usd:.2f} USD/{price_info['period']}</b>\n\n"
        "Selecciona tu metodo de pago:",
        parse_mode="HTML",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_pay_stripe(update, context):
    """Handle Stripe payment — create checkout session."""
    import stripe

    query = update.callback_query
    await query.answer()

    plan_key = query.data.replace("pay_stripe_", "")  # e.g. "basic_monthly"
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    stripe_key = os.getenv("STRIPE_SECRET_KEY")
    if not stripe_key:
        await query.message.reply_text(
            "Stripe no disponible temporalmente.\n"
            "Intenta con Mercado Pago o contacta /soporte."
        )
        return

    stripe.api_key = stripe_key

    price_ids = {
        "basic_monthly": os.getenv("STRIPE_PRICE_BASIC"),
        "basic_annual": os.getenv("STRIPE_PRICE_BASIC_ANNUAL"),
        "premium_monthly": os.getenv("STRIPE_PRICE_PREMIUM"),
        "premium_annual": os.getenv("STRIPE_PRICE_PREMIUM_ANNUAL"),
    }

    price_id = price_ids.get(plan_key)
    if not price_id:
        await query.message.reply_text(
            "Este plan aun no esta disponible en Stripe.\n"
            "Intenta con Mercado Pago o contacta /soporte."
        )
        return

    user = query.from_user
    plan_base = plan_key.split("_")[0]  # "basic" or "premium"
    bot_username = (await context.bot.get_me()).username

    # Promo discount
    promo = context.user_data.get("promo")
    d_usd, _, promo_id = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])

    try:
        session_params = {
            "payment_method_types": ["card"],
            "line_items": [{"price": price_id, "quantity": 1}],
            "mode": "subscription",
            "success_url": f"https://t.me/{bot_username}?start=payment_success",
            "cancel_url": f"https://t.me/{bot_username}?start=payment_cancel",
            "metadata": {
                "telegram_id": str(user.id),
                "plan": plan_base,
                "period": plan_key,
            },
            "client_reference_id": str(user.id),
        }

        if promo_id:
            coupon = stripe.Coupon.create(
                percent_off=promo["discount_percent"] if promo.get("discount_percent") else None,
                amount_off=int(promo["discount_amount_usd"] * 100) if promo.get("discount_amount_usd") else None,
                currency="usd" if promo.get("discount_amount_usd") else None,
                duration="once",
                name=f"Promo {promo['code']}",
            )
            session_params["discounts"] = [{"coupon": coupon.id}]
            session_params["metadata"]["promo_id"] = str(promo_id)

        session = stripe.checkout.Session.create(**session_params)

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [[InlineKeyboardButton("Pagar con Stripe", url=session.url)]]

        show_usd = d_usd if promo_id else price_info["usd"]
        await query.message.reply_text(
            f"<b>Stripe - {price_info['label']} (${show_usd:.2f} USD/{price_info['period']})</b>\n\n"
            "Haz clic para completar el pago:\n"
            "(Tarjetas de credito y debito)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        logger.info(f"Stripe checkout for {user.id} ({plan_key}){' promo=' + promo['code'] if promo_id else ''}")

    except Exception as e:
        logger.error(f"Stripe error: {e}")
        await query.message.reply_text("Error con Stripe. Intenta Mercado Pago o /soporte.")


async def handle_pay_mp(update, context):
    """Handle Mercado Pago payment — create subscription (preapproval)."""
    import mercadopago

    query = update.callback_query
    await query.answer()

    plan_key = query.data.replace("pay_mp_", "")  # e.g. "basic_monthly"
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    mp_token = os.getenv("MP_ACCESS_TOKEN")
    if not mp_token:
        await query.message.reply_text(
            "Mercado Pago no disponible temporalmente.\n"
            "Intenta con Stripe o contacta /soporte."
        )
        return

    sdk = mercadopago.SDK(mp_token)
    user = query.from_user
    plan_base = plan_key.split("_")[0]  # "basic" or "premium"
    is_annual = "annual" in plan_key
    bot_username = (await context.bot.get_me()).username

    # Get user email (required by MP)
    bot_user = db.get_bot_user(user.id)
    user_email = bot_user.get("email") if bot_user else None
    if not user_email:
        await query.message.reply_text(
            "Necesito tu email para Mercado Pago.\n"
            "Escribe tu email y luego vuelve a /suscribir."
        )
        context.user_data["awaiting_email"] = True
        return

    # Promo discount
    promo = context.user_data.get("promo")
    _, d_mxn, promo_id = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])
    mxn_amount = float(d_mxn) if promo_id else float(price_info["mxn"])
    ext_ref = f"{user.id}_{plan_base}_{promo_id}" if promo_id else f"{user.id}_{plan_base}"

    preapproval_data = {
        "reason": f"MedExpert {price_info['label']}",
        "payer_email": user_email,
        "auto_recurring": {
            "frequency": 12 if is_annual else 1,
            "frequency_type": "months",
            "transaction_amount": mxn_amount,
            "currency_id": "MXN",
        },
        "back_url": f"https://t.me/{bot_username}?start=payment_success",
        "external_reference": ext_ref,
        "notification_url": "https://api.medexpert.mx/api/mercadopago/webhook",
    }

    try:
        result = sdk.preapproval().create(preapproval_data)
        response = result.get("response", {})
        checkout_url = response.get("sandbox_init_point") or response.get("init_point")

        if not checkout_url:
            await query.message.reply_text("Error creando suscripcion. Contacta /soporte.")
            logger.error(f"MP preapproval error: {result}")
            return

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [[InlineKeyboardButton("Pagar con Mercado Pago", url=checkout_url)]]

        await query.message.reply_text(
            f"<b>Mercado Pago - {price_info['label']} (${mxn_amount:.0f} MXN/{price_info['period']})</b>\n\n"
            "Haz clic para suscribirte:\n"
            "(Tarjeta, transferencia, OXXO y mas)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        logger.info(f"MP preapproval for {user.id} ({plan_key}){' promo=' + promo['code'] if promo_id else ''}")

    except Exception as e:
        logger.error(f"MercadoPago error: {e}")
        await query.message.reply_text("Error con Mercado Pago. Intenta Stripe o /soporte.")


async def handle_pay_paypal(update, context):
    """Handle PayPal payment — create subscription via REST API."""
    import httpx

    query = update.callback_query
    await query.answer()

    plan_key = query.data.replace("pay_paypal_", "")
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    # PayPal uses fixed plan_ids — no price override possible
    promo = context.user_data.get("promo")
    _, _, promo_id = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])
    if promo_id:
        await query.message.reply_text(
            "Los codigos promocionales no estan disponibles para PayPal.\n\n"
            "Usa <b>Stripe</b> (tarjeta) para aplicar tu descuento,\n"
            "o selecciona otra forma de pago.",
            parse_mode="HTML",
        )
        return

    client_id = os.getenv("PAYPAL_CLIENT_ID")
    client_secret = os.getenv("PAYPAL_CLIENT_SECRET")
    if not client_id or not client_secret:
        await query.message.reply_text(
            "PayPal no disponible temporalmente.\n"
            "Intenta con Stripe o Mercado Pago."
        )
        return

    plan_ids = {
        "basic_monthly": os.getenv("PAYPAL_PLAN_BASIC"),
        "basic_annual": os.getenv("PAYPAL_PLAN_BASIC_ANNUAL"),
        "premium_monthly": os.getenv("PAYPAL_PLAN_PREMIUM"),
        "premium_annual": os.getenv("PAYPAL_PLAN_PREMIUM_ANNUAL"),
    }

    paypal_plan_id = plan_ids.get(plan_key)
    if not paypal_plan_id:
        await query.message.reply_text(
            "Este plan aun no esta disponible en PayPal.\n"
            "Intenta con Stripe o Mercado Pago."
        )
        return

    user = query.from_user
    plan_base = plan_key.split("_")[0]
    bot_username = (await context.bot.get_me()).username
    paypal_base = os.getenv("PAYPAL_API_BASE", "https://api-m.sandbox.paypal.com")

    try:
        async with httpx.AsyncClient() as client:
            # Get OAuth2 token
            token_resp = await client.post(
                f"{paypal_base}/v1/oauth2/token",
                auth=(client_id, client_secret),
                data={"grant_type": "client_credentials"},
            )
            access_token = token_resp.json().get("access_token")

            # Create subscription
            sub_resp = await client.post(
                f"{paypal_base}/v1/billing/subscriptions",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "plan_id": paypal_plan_id,
                    "custom_id": f"{user.id}_{plan_base}",
                    "application_context": {
                        "return_url": f"https://t.me/{bot_username}?start=payment_success",
                        "cancel_url": f"https://t.me/{bot_username}?start=payment_cancel",
                        "brand_name": "MedExpert",
                        "user_action": "SUBSCRIBE_NOW",
                    },
                },
            )
            sub_data = sub_resp.json()

            # Find approval link
            approve_url = None
            for link in sub_data.get("links", []):
                if link.get("rel") == "approve":
                    approve_url = link["href"]
                    break

            if not approve_url:
                await query.message.reply_text("Error creando suscripcion PayPal. Contacta /soporte.")
                logger.error(f"PayPal subscription error: {sub_data}")
                return

            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [[InlineKeyboardButton("Pagar con PayPal", url=approve_url)]]

            await query.message.reply_text(
                f"<b>PayPal - {price_info['label']} (${price_info['usd']:.2f} USD/{price_info['period']})</b>\n\n"
                "Haz clic para suscribirte con PayPal:",
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            logger.info(f"PayPal subscription for {user.id} ({plan_key})")

    except Exception as e:
        logger.error(f"PayPal error: {e}")
        await query.message.reply_text("Error con PayPal. Intenta Stripe o Mercado Pago.")


async def handle_pay_clip(update, context):
    """Handle Clip payment — create checkout payment link (MXN)."""
    import httpx
    import base64

    query = update.callback_query
    await query.answer()

    plan_key = query.data.replace("pay_clip_", "")  # e.g. "basic_monthly"
    price_info = PLAN_PRICES.get(plan_key)
    if not price_info:
        await query.message.reply_text("Plan no disponible.")
        return

    clip_api_key = os.getenv("CLIP_API_KEY")
    clip_api_secret = os.getenv("CLIP_API_SECRET")
    if not clip_api_key or not clip_api_secret:
        await query.message.reply_text(
            "Clip no disponible temporalmente.\n"
            "Intenta con Mercado Pago o Stripe."
        )
        return

    user = query.from_user
    plan_base = plan_key.split("_")[0]  # "basic" or "premium"
    bot_username = (await context.bot.get_me()).username
    credentials = base64.b64encode(f"{clip_api_key}:{clip_api_secret}".encode()).decode()

    # Promo discount
    promo = context.user_data.get("promo")
    _, d_mxn, promo_id = apply_promo_discount(promo, plan_key, price_info["usd"], price_info["mxn"])
    mxn_amount = float(d_mxn) if promo_id else float(price_info["mxn"])
    ext_ref = f"{user.id}_{plan_base}_{promo_id}" if promo_id else f"{user.id}_{plan_base}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.payclip.com/v2/checkout",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                },
                json={
                    "amount": mxn_amount,
                    "currency": "MXN",
                    "purchase_description": f"MedExpert {price_info['label']}",
                    "redirection_url": {
                        "success": f"https://t.me/{bot_username}?start=payment_success",
                        "error": f"https://t.me/{bot_username}?start=payment_cancel",
                        "default": f"https://t.me/{bot_username}",
                    },
                    "metadata": {
                        "external_reference": ext_ref,
                    },
                    "custom_payment_options": {
                        "payment_method_types": ["debit", "credit", "cash"],
                    },
                    "webhook_url": "https://api.medexpert.mx/api/clip/webhook",
                },
                timeout=15.0,
            )

        data = resp.json()
        checkout_url = data.get("payment_request_url")

        if not checkout_url:
            await query.message.reply_text("Error creando pago con Clip. Contacta /soporte.")
            logger.error(f"Clip checkout error: {data}")
            return

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [[InlineKeyboardButton("Pagar con Clip", url=checkout_url)]]

        await query.message.reply_text(
            f"<b>Clip - {price_info['label']} (${mxn_amount:.0f} MXN/{price_info['period']})</b>\n\n"
            "Haz clic para completar el pago:\n"
            "(Tarjeta, OXXO)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        logger.info(f"Clip checkout for {user.id} ({plan_key}): {data.get('payment_request_id')}{' promo=' + promo['code'] if promo_id else ''}")

    except Exception as e:
        logger.error(f"Clip error: {e}")
        await query.message.reply_text("Error con Clip. Intenta Mercado Pago o Stripe.")


# ─────────────────────────────────────────────
# Source Preferences (/fuentes)
# ─────────────────────────────────────────────

def _build_source_filter(telegram_id: int) -> dict | None:
    """Build ChromaDB where filter from user's source preferences."""
    user_sources = db.get_bot_user_sources(telegram_id)
    if user_sources:
        return {"society": {"$in": user_sources}}
    return None


def _fuentes_keyboard(enabled: list[str] | None):
    """Build inline keyboard for source toggles."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    active = set(enabled) if enabled else set(_ALL_SOURCES)
    rows = []
    for src in _ALL_SOURCES:
        icon = "✅" if src in active else "⬜"
        rows.append([InlineKeyboardButton(f"{icon} {src}", callback_data=f"src_{src}")])
    rows.append([InlineKeyboardButton("Restablecer todas", callback_data="src_reset")])
    return InlineKeyboardMarkup(rows)


def _fuentes_text(enabled: list[str] | None) -> str:
    """Build display text for current source preferences."""
    active = set(enabled) if enabled else set(_ALL_SOURCES)
    lines = ["📚 Tus fuentes activas:\n"]
    for src in _ALL_SOURCES:
        icon = "✅" if src in active else "⬜"
        lines.append(f"{icon} {_SOURCE_LABELS[src]}")
    lines.append("\nToca una fuente para activar/desactivar.")
    return "\n".join(lines)


async def cmd_fuentes(update, context):
    """Show source preference toggles."""
    user = update.effective_user
    db.get_or_create_bot_user(
        telegram_id=user.id,
        username=user.username or "",
        first_name=user.first_name or "",
        specialty=context.bot_data.get("specialty", "oncologia"),
    )
    enabled = db.get_bot_user_sources(user.id)
    await update.message.reply_text(
        _fuentes_text(enabled),
        reply_markup=_fuentes_keyboard(enabled),
    )


async def handle_source_toggle(update, context):
    """Toggle a single source on/off."""
    query = update.callback_query
    await query.answer()
    src = query.data.replace("src_", "")
    if src not in _ALL_SOURCES:
        return

    user_id = query.from_user.id
    enabled = db.get_bot_user_sources(user_id)
    active = list(enabled) if enabled else list(_ALL_SOURCES)

    if src in active:
        if len(active) <= 1:
            await query.answer("Debes tener al menos 1 fuente activa.", show_alert=True)
            return
        active.remove(src)
    else:
        active.append(src)

    # If all enabled, store NULL (default)
    if set(active) == set(_ALL_SOURCES):
        db.update_bot_user(user_id, source_preferences_json=None)
        enabled_new = None
    else:
        db.update_bot_user(user_id, source_preferences_json=_json.dumps(active))
        enabled_new = active

    try:
        await query.edit_message_text(
            _fuentes_text(enabled_new),
            reply_markup=_fuentes_keyboard(enabled_new),
        )
    except Exception:
        pass  # Message unchanged


async def handle_source_reset(update, context):
    """Reset all sources to enabled (default)."""
    query = update.callback_query
    user_id = query.from_user.id
    db.update_bot_user(user_id, source_preferences_json=None)
    try:
        await query.edit_message_text(
            _fuentes_text(None),
            reply_markup=_fuentes_keyboard(None),
        )
    except Exception:
        pass  # Already showing all sources
    await query.answer("Fuentes restablecidas")


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

        # Query RAG + LLM (with user source filter)
        brain = get_brain(specialty)
        source_filter = _build_source_filter(user.id)
        result = brain.query(transcript, expert_slug=specialty, source_filter=source_filter)

        # Extract clinical metadata
        metadata = extract_clinical_metadata(transcript, result.get("response", ""), specialty)
        metadata_json = _json.dumps(metadata, ensure_ascii=False)

        # Log consultation
        user_plan = db.get_bot_user_plan(user.id)
        free_used = db.count_bot_free_queries(user.id, specialty)
        is_free = user_plan == "free" and free_used < FREE_QUERY_LIMIT
        consultation_id = db.log_bot_consultation(
            telegram_id=user.id,
            specialty=specialty,
            query_type="voice",
            query_text=anonymize_text(transcript),
            response_text=result.get("response", ""),
            response_time_seconds=result.get("processing_time", 0),
            llm_provider=result.get("provider", ""),
            llm_model=result.get("model", ""),
            tokens_input=result.get("token_usage", {}).get("input_tokens", 0),
            tokens_output=result.get("token_usage", {}).get("output_tokens", 0),
            rag_chunks_used=result.get("rag_chunks_used", 0),
            is_free_tier=is_free,
            citations=result.get("citations", []),
            clinical_metadata_json=metadata_json,
        )

        # Format and send response
        free_remaining = max(0, FREE_QUERY_LIMIT - free_used - 1)
        show_free = free_remaining if (user_plan == "free" and is_free) else None
        main_text, footer = format_response_for_telegram(
            result,
            free_remaining=show_free,
        )

        # Delete processing message and send response
        await processing_msg.delete()

        # Send main response as plain text (LLM output may have Markdown)
        await _send_long_message(update, main_text)

        # Send footer (citations, free tier) with action buttons
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [
                    InlineKeyboardButton("Profundizar", callback_data=f"deepen_{consultation_id}"),
                    InlineKeyboardButton("Exportar PDF", callback_data=f"pdf_{consultation_id}"),
                ],
                [
                    InlineKeyboardButton("¿Te sirvió?", callback_data=f"eval_{consultation_id}"),
                ],
            ]
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
    import re

    user = update.effective_user
    specialty = context.bot_data.get("specialty", "oncologia")
    text = update.message.text.strip()

    if not text:
        return

    # Intercept pending deepen follow-up question
    if context.user_data.get("pending_deepen"):
        consultation_id = context.user_data.pop("pending_deepen")
        await _handle_deepen_with_question(update, context, consultation_id, text)
        return

    # Intercept email capture
    if context.user_data.get("awaiting_email"):
        email = text.strip().lower()
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            db.update_bot_user(user.id, email=email)
            context.user_data.pop("awaiting_email", None)
            await update.message.reply_text(
                f"Email guardado: <b>{email}</b>\n\n"
                "Ahora usa /suscribir para ver los planes.",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(
                "Ese no parece un email valido. Intenta de nuevo:\n"
                "(ejemplo: doctor@hospital.com)"
            )
        return

    # Intercept support ticket creation
    if context.user_data.get("awaiting_support"):
        context.user_data.pop("awaiting_support", None)
        ticket_id = db.create_ticket(
            telegram_id=user.id,
            ticket_type="support",
            title=text[:80],
            description=text,
            expert_slug=specialty,
        )
        logger.info(f"Support ticket #{ticket_id} created by {user.id}")
        await update.message.reply_text(
            f"<b>Ticket #{ticket_id} creado</b>\n\n"
            "Tu mensaje fue enviado a nuestro equipo de soporte. "
            "Te notificaremos cuando tengamos una respuesta.",
            parse_mode="HTML",
        )
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

        # Query RAG + LLM (with user source filter)
        brain = get_brain(specialty)
        source_filter = _build_source_filter(user.id)
        result = brain.query(text, expert_slug=specialty, source_filter=source_filter)

        # Extract clinical metadata
        metadata = extract_clinical_metadata(text, result.get("response", ""), specialty)
        metadata_json = _json.dumps(metadata, ensure_ascii=False)

        # Log consultation
        user_plan = db.get_bot_user_plan(user.id)
        free_used = db.count_bot_free_queries(user.id, specialty)
        is_free = user_plan == "free" and free_used < FREE_QUERY_LIMIT
        consultation_id = db.log_bot_consultation(
            telegram_id=user.id,
            specialty=specialty,
            query_type="text",
            query_text=anonymize_text(text),
            response_text=result.get("response", ""),
            response_time_seconds=result.get("processing_time", 0),
            llm_provider=result.get("provider", ""),
            llm_model=result.get("model", ""),
            tokens_input=result.get("token_usage", {}).get("input_tokens", 0),
            tokens_output=result.get("token_usage", {}).get("output_tokens", 0),
            rag_chunks_used=result.get("rag_chunks_used", 0),
            is_free_tier=is_free,
            citations=result.get("citations", []),
            clinical_metadata_json=metadata_json,
        )

        # Format and send response
        free_remaining = max(0, FREE_QUERY_LIMIT - free_used - 1)
        show_free = free_remaining if (user_plan == "free" and is_free) else None
        main_text, footer = format_response_for_telegram(
            result,
            free_remaining=show_free,
        )

        # Delete processing message and send response
        await processing_msg.delete()

        # Send main response as plain text (LLM output may have Markdown)
        await _send_long_message(update, main_text)

        # Send footer (citations, free tier) with action buttons
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [
                    InlineKeyboardButton("Profundizar", callback_data=f"deepen_{consultation_id}"),
                    InlineKeyboardButton("Exportar PDF", callback_data=f"pdf_{consultation_id}"),
                ],
                [
                    InlineKeyboardButton("¿Te sirvió?", callback_data=f"eval_{consultation_id}"),
                ],
            ]
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
        "<b>Plan Basico - $14.99 USD/mes</b> (precio de lanzamiento)\n"
        "  Consultas ilimitadas\n"
        "  Profundizar con GPT-OSS 120B\n"
        "  Respuesta en menos de 5 segundos\n"
        "  Cancela cuando quieras\n\n"
        "<b>Plan Premium - $24.99 USD/mes</b> (precio de lanzamiento)\n"
        "  Todo lo del Plan Basico\n"
        "  Profundizar con Claude Opus 4.6\n"
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
    """Handle 'Profundizar' button — show choice: general deepen or ask a question."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

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

    if consultation.get("is_deepening"):
        await query.message.reply_text("Esta consulta ya fue profundizada.")
        return

    keyboard = [
        [
            InlineKeyboardButton("Profundizar en general", callback_data=f"deepen_go_{consultation_id}"),
            InlineKeyboardButton("Tengo una pregunta", callback_data=f"deepen_ask_{consultation_id}"),
        ],
    ]
    await query.message.reply_text(
        "¿Quieres profundizar en algo especifico?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_deepen_ask(update, context):
    """Handle 'Tengo una pregunta' — store pending state, ask user to type question."""
    query = update.callback_query
    await query.answer()

    consultation_id = int(query.data.split("_")[2])
    consultation = db.get_bot_consultation_by_id(consultation_id)
    if not consultation:
        await query.message.reply_text("Consulta no encontrada.")
        return

    context.user_data["pending_deepen"] = consultation_id
    await query.message.reply_text(
        "Escribe tu pregunta de seguimiento:\n"
        "(Ej: ¿Qué efectos adversos tiene el pembrolizumab?)\n\n"
        "Usa /cancelar para cancelar."
    )


async def _handle_deepen_with_question(update, context, consultation_id: int, followup_question: str):
    """Execute deepen with a specific follow-up question."""
    await _execute_deepen(update, context, consultation_id, followup_question=followup_question)


async def handle_deepen_go(update, context):
    """Handle 'Profundizar en general' — immediate deepen."""
    query = update.callback_query
    await query.answer()

    consultation_id = int(query.data.split("_")[2])
    await _execute_deepen(query, context, consultation_id, is_callback=True)


async def _execute_deepen(update_or_query, context, consultation_id: int,
                          followup_question: str = None, is_callback: bool = False):
    """Core deepen logic shared by general deepen and follow-up question deepen.

    Free + basic plan: GPT-OSS 120B via Groq (~3s)
    Premium plan: configured deepen model (~5-30s)
    """
    consultation = db.get_bot_consultation_by_id(consultation_id)
    if not consultation:
        msg = "Consulta no encontrada."
        if is_callback:
            await update_or_query.message.reply_text(msg)
        else:
            await update_or_query.message.reply_text(msg)
        return

    if is_callback:
        user_id = update_or_query.from_user.id
    else:
        user_id = update_or_query.effective_user.id

    if consultation["telegram_id"] != user_id:
        msg = "No tienes acceso a esta consulta."
        if is_callback:
            await update_or_query.message.reply_text(msg)
        else:
            await update_or_query.message.reply_text(msg)
        return

    specialty = consultation.get("specialty", "oncologia")

    if consultation.get("is_deepening"):
        msg = "Esta consulta ya fue profundizada."
        if is_callback:
            await update_or_query.message.reply_text(msg)
        else:
            await update_or_query.message.reply_text(msg)
        return

    # Determine user tier
    user_plan = db.get_bot_user_plan(user_id)

    if user_plan == "premium":
        opus_today = db.count_bot_opus_deepenings_today(user_id, specialty)
        if opus_today >= 3:
            msg = ("Ya usaste tus 3 profundizaciones premium de hoy.\n"
                   "Se renuevan manana. Puedes seguir consultando normalmente.")
            if is_callback:
                await update_or_query.message.reply_text(msg)
            else:
                await update_or_query.message.reply_text(msg)
            return
        tier = "premium"
    elif user_plan == "basic":
        sonnet_today = db.count_bot_sonnet_deepenings_today(user_id, specialty)
        if sonnet_today >= 5:
            msg = ("Ya usaste tus 5 profundizaciones de hoy.\n"
                   "Se renuevan manana.\n\n"
                   "Plan Plus ($24.99 USD/mes): profundizar con modelo avanzado\n"
                   "/suscribir para cambiar de plan")
            if is_callback:
                await update_or_query.message.reply_text(msg)
            else:
                await update_or_query.message.reply_text(msg)
            return
        tier = "basic"
    else:
        tier = "free"

    free_used = db.count_bot_free_queries(user_id, specialty)
    is_free = user_plan == "free"

    if is_free:
        if not db.can_bot_user_query(user_id, specialty, FREE_QUERY_LIMIT):
            msg = ("No tienes consultas gratis restantes.\n"
                   "Profundizar consume 1 consulta.\n\n"
                   "/suscribir para activar")
            if is_callback:
                await update_or_query.message.reply_text(msg)
            else:
                await update_or_query.message.reply_text(msg)
            return

    # Build model label from config
    config = db.get_expert_llm_config(specialty)
    if tier in ("premium", "basic"):
        model_label = config["deepen_model"]
    else:
        model_label = config["base_model"]

    # Show processing message
    extra = " con tu pregunta" if followup_question else ""
    if is_callback:
        processing_msg = await update_or_query.message.reply_text(
            f"Profundizando{extra} con {model_label}..."
        )
    else:
        processing_msg = await update_or_query.message.reply_text(
            f"Profundizando{extra} con {model_label}..."
        )

    try:
        brain = get_brain(specialty)
        source_filter = _build_source_filter(user_id)
        result = brain.deepen(
            original_query=consultation["query_text"],
            original_response=consultation["response_text"],
            expert_slug=specialty,
            tier=tier,
            source_filter=source_filter,
            followup_question=followup_question,
        )

        # Extract clinical metadata from deepened response
        metadata = extract_clinical_metadata(
            consultation["query_text"], result.get("response", ""), specialty
        )
        metadata_json = _json.dumps(metadata, ensure_ascii=False)

        # Log as consultation
        deepen_id = db.log_bot_consultation(
            telegram_id=user_id,
            specialty=specialty,
            query_type=consultation.get("query_type", "text"),
            query_text=followup_question or consultation["query_text"],
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
            clinical_metadata_json=metadata_json,
        )

        # Format and send
        main_text, footer = format_response_for_telegram(result)

        await processing_msg.delete()

        # Send deepened response
        if is_callback:
            await _send_long_message_from_callback(update_or_query, main_text)
        else:
            await _send_long_message(update_or_query, main_text)

        # Send footer with PDF + feedback buttons
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [InlineKeyboardButton("Exportar PDF", callback_data=f"pdf_{deepen_id}")],
                [InlineKeyboardButton("¿Te sirvió?", callback_data=f"eval_{deepen_id}")],
            ]
            if is_callback:
                await update_or_query.message.reply_text(
                    footer, parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )
            else:
                await update_or_query.message.reply_text(
                    footer, parse_mode="HTML",
                    reply_markup=InlineKeyboardMarkup(keyboard),
                )

        logger.info(f"Deepen for {user_id} (tier={tier}, model={model_label})"
                     f"{' followup' if followup_question else ''}")

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


FEEDBACK_OPTIONS = {
    "fb_a": "Mejoró mi manera de ver las cosas",
    "fb_b": "Reforzó mi plan",
    "fb_c": "Información incorrecta",
    "fb_d": "Información incompleta",
    "fb_e": "No me sirvió (otra razón)",
}


async def handle_feedback_prompt(update, context):
    """Handle 'Evaluar' button — show feedback options."""
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data.startswith("eval_"):
        return

    consultation_id = int(data.split("_")[1])

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton(f"a) {FEEDBACK_OPTIONS['fb_a']}", callback_data=f"fb_a_{consultation_id}")],
        [InlineKeyboardButton(f"b) {FEEDBACK_OPTIONS['fb_b']}", callback_data=f"fb_b_{consultation_id}")],
        [InlineKeyboardButton(f"c) {FEEDBACK_OPTIONS['fb_c']}", callback_data=f"fb_c_{consultation_id}")],
        [InlineKeyboardButton(f"d) {FEEDBACK_OPTIONS['fb_d']}", callback_data=f"fb_d_{consultation_id}")],
        [InlineKeyboardButton(f"e) {FEEDBACK_OPTIONS['fb_e']}", callback_data=f"fb_e_{consultation_id}")],
    ]
    await query.message.reply_text(
        "¿La respuesta te sirvió?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


async def handle_feedback_response(update, context):
    """Handle feedback selection — save to DB."""
    query = update.callback_query
    await query.answer()

    data = query.data  # e.g. "fb_a_123"
    parts = data.split("_")
    feedback_key = f"{parts[0]}_{parts[1]}"  # "fb_a"
    consultation_id = int(parts[2])

    feedback_text = FEEDBACK_OPTIONS.get(feedback_key, "Desconocido")
    db.update_bot_consultation_feedback(consultation_id, feedback_text)

    # Replace feedback buttons with thank you
    try:
        await query.edit_message_text(f"Gracias por tu feedback: {feedback_text}")
    except Exception:
        pass

    logger.info(f"Feedback for consultation {consultation_id}: {feedback_text}")


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
    app.add_handler(CommandHandler("cancelar", cmd_cancelar))
    app.add_handler(CommandHandler("suscribir", cmd_suscribir))
    app.add_handler(CommandHandler("codigo", cmd_codigo))
    app.add_handler(CommandHandler("verificar", cmd_verificar))
    app.add_handler(CommandHandler("congresos", cmd_congresos))
    app.add_handler(CommandHandler("fuentes", cmd_fuentes))

    # Photo/document handler for verification flow
    async def handle_photo_or_doc(update, context):
        handled = await handle_verification_photo(update, context)
        if not handled:
            await update.message.reply_text(
                "Envía un mensaje de texto o audio con tu consulta clínica."
            )
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.ALL, handle_photo_or_doc))

    # Message handlers (voice and text)
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Callback handlers (PDF export button)
    from telegram.ext import CallbackQueryHandler
    app.add_handler(CallbackQueryHandler(handle_deepen_callback, pattern=r"^deepen_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_deepen_go, pattern=r"^deepen_go_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_deepen_ask, pattern=r"^deepen_ask_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_pdf_callback, pattern=r"^pdf_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_feedback_prompt, pattern=r"^eval_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_feedback_response, pattern=r"^fb_[a-e]_\d+$"))
    app.add_handler(CallbackQueryHandler(handle_subscribe_callback, pattern=r"^subscribe_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_region_mx, pattern=r"^region_mx_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_region_intl, pattern=r"^region_intl_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_pay_stripe, pattern=r"^pay_stripe_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_pay_mp, pattern=r"^pay_mp_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_pay_paypal, pattern=r"^pay_paypal_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_pay_clip, pattern=r"^pay_clip_(basic|premium)_(monthly|annual)$"))
    app.add_handler(CallbackQueryHandler(handle_source_toggle, pattern=r"^src_(NCCN|ESMO|NCI|IMSS)$"))
    app.add_handler(CallbackQueryHandler(handle_source_reset, pattern=r"^src_reset$"))

    # Error handler
    app.add_error_handler(handle_error)

    # Set bot commands menu (visible when user types "/")
    async def post_init(application):
        from telegram import BotCommand
        await application.bot.set_my_commands([
            BotCommand("start", "Iniciar el bot"),
            BotCommand("ayuda", "Ver comandos disponibles"),
            BotCommand("estado", "Ver tu plan y consultas"),
            BotCommand("suscribir", "Ver planes de suscripcion"),
            BotCommand("codigo", "Aplicar codigo promocional"),
            BotCommand("verificar", "Verificar cedula profesional"),
            BotCommand("congresos", "Proximos congresos medicos"),
            BotCommand("fuentes", "Elegir fuentes de guias clinicas"),
            BotCommand("soporte", "Contactar soporte"),
            BotCommand("terminos", "Terminos y condiciones"),
            BotCommand("cancelar", "Cancelar accion en curso"),
        ])
        logger.info("Bot commands menu set")
    app.post_init = post_init

    # Start
    logger.info(f"MedExpert Bot starting (specialty: {specialty}, polling mode)")
    logger.info("Send /start to the bot on Telegram to begin")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
