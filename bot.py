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

    # Check args (referral, payment redirect)
    if context.args:
        arg = context.args[0]
        if arg == "payment_success":
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
    for key in ("awaiting_support", "awaiting_email", "awaiting_verification"):
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

    keyboard = [
        [InlineKeyboardButton(
            "Basico - $14.99 USD/mes",
            callback_data="subscribe_basic_monthly",
        )],
        [InlineKeyboardButton(
            "Basico - $149.90 USD/año (2 meses gratis)",
            callback_data="subscribe_basic_annual",
        )],
        [InlineKeyboardButton(
            "Premium - $24.99 USD/mes",
            callback_data="subscribe_premium_monthly",
        )],
        [InlineKeyboardButton(
            "Premium - $249.90 USD/año (2 meses gratis)",
            callback_data="subscribe_premium_annual",
        )],
    ]

    await update.message.reply_text(
        "<b>Suscripciones MedExpert</b>\n\n"
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

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton(
            f"Mexico - ${price_info['mxn']:.0f} MXN",
            callback_data=f"region_mx_{plan_key}",
        )],
        [InlineKeyboardButton(
            f"Internacional - ${price_info['usd']:.2f} USD",
            callback_data=f"region_intl_{plan_key}",
        )],
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

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton(
            f"Mercado Pago - ${price_info['mxn']:.0f} MXN",
            callback_data=f"pay_mp_{plan_key}",
        )],
        [InlineKeyboardButton(
            f"Clip (OXXO/Tarjeta) - ${price_info['mxn']:.0f} MXN",
            callback_data=f"pay_clip_{plan_key}",
        )],
    ]

    await query.message.reply_text(
        f"<b>{price_info['label']} - ${price_info['mxn']:.0f} MXN/{price_info['period']}</b>\n\n"
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

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    keyboard = [
        [InlineKeyboardButton(
            f"Tarjeta (Stripe) - ${price_info['usd']:.2f} USD",
            callback_data=f"pay_stripe_{plan_key}",
        )],
        [InlineKeyboardButton(
            f"PayPal - ${price_info['usd']:.2f} USD",
            callback_data=f"pay_paypal_{plan_key}",
        )],
    ]

    await query.message.reply_text(
        f"<b>{price_info['label']} - ${price_info['usd']:.2f} USD/{price_info['period']}</b>\n\n"
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

    try:
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=f"https://t.me/{bot_username}?start=payment_success",
            cancel_url=f"https://t.me/{bot_username}?start=payment_cancel",
            metadata={"telegram_id": str(user.id), "plan": plan_base, "period": plan_key},
            client_reference_id=str(user.id),
        )

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup
        keyboard = [[InlineKeyboardButton("Pagar con Stripe", url=session.url)]]

        await query.message.reply_text(
            f"<b>Stripe - {price_info['label']} (${price_info['usd']:.2f} USD/{price_info['period']})</b>\n\n"
            "Haz clic para completar el pago:\n"
            "(Tarjetas de credito y debito)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        logger.info(f"Stripe checkout for {user.id} ({plan_key})")

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

    preapproval_data = {
        "reason": f"MedExpert {price_info['label']}",
        "payer_email": user_email,
        "auto_recurring": {
            "frequency": 12 if is_annual else 1,
            "frequency_type": "months",
            "transaction_amount": float(price_info["mxn"]),
            "currency_id": "MXN",
        },
        "back_url": f"https://t.me/{bot_username}?start=payment_success",
        "external_reference": f"{user.id}_{plan_base}",
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
            f"<b>Mercado Pago - {price_info['label']} (${price_info['mxn']} MXN/{price_info['period']})</b>\n\n"
            "Haz clic para suscribirte:\n"
            "(Tarjeta, transferencia, OXXO y mas)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        logger.info(f"MP preapproval for {user.id} ({plan_key})")

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

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.payclip.com/v2/checkout",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                },
                json={
                    "amount": float(price_info["mxn"]),
                    "currency": "MXN",
                    "purchase_description": f"MedExpert {price_info['label']}",
                    "redirection_url": {
                        "success": f"https://t.me/{bot_username}?start=payment_success",
                        "error": f"https://t.me/{bot_username}?start=payment_cancel",
                        "default": f"https://t.me/{bot_username}",
                    },
                    "metadata": {
                        "external_reference": f"{user.id}_{plan_base}",
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
            f"<b>Clip - {price_info['label']} (${price_info['mxn']:.0f} MXN/{price_info['period']})</b>\n\n"
            "Haz clic para completar el pago:\n"
            "(Tarjeta, OXXO)",
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )
        logger.info(f"Clip checkout for {user.id} ({plan_key}): {data.get('payment_request_id')}")

    except Exception as e:
        logger.error(f"Clip error: {e}")
        await query.message.reply_text("Error con Clip. Intenta Mercado Pago o Stripe.")


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
        user_plan = db.get_bot_user_plan(user.id)
        free_used = db.count_bot_free_queries(user.id, specialty)
        is_free = user_plan == "free" and free_used < FREE_QUERY_LIMIT
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

        # Query RAG + LLM
        brain = get_brain()
        result = brain.query(text, expert_slug=specialty)

        # Log consultation
        user_plan = db.get_bot_user_plan(user.id)
        free_used = db.count_bot_free_queries(user.id, specialty)
        is_free = user_plan == "free" and free_used < FREE_QUERY_LIMIT
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
    bot_user = db.get_bot_user(user_id)
    user_plan = db.get_bot_user_plan(user_id)

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
                "Plan Basico ($14.99 USD/mes): consultas ilimitadas + profundizar con GPT-OSS 120B\n"
                "Plan Premium ($24.99 USD/mes): todo + profundizar con Claude Opus 4.6\n\n"
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

        # Send footer with PDF + feedback buttons (no deepen button on deepened response)
        if footer:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = [
                [InlineKeyboardButton("Exportar PDF", callback_data=f"pdf_{deepen_id}")],
                [InlineKeyboardButton("¿Te sirvió?", callback_data=f"eval_{deepen_id}")],
            ]
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
    app.add_handler(CommandHandler("verificar", cmd_verificar))

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
            BotCommand("verificar", "Verificar cedula profesional"),
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
