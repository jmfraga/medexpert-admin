#!/bin/bash
# Health check for MedExpert services — sends Telegram alert if down
# Cron: */5 * * * * /Users/juanma/Projects/medexpert-admin/scripts/health_check.sh

# Load bot token and admin chat ID from .env
ENV_FILE="/Users/juanma/Projects/medexpert-admin/.env"
BOT_TOKEN=$(grep '^TELEGRAM_BOT_TOKEN=' "$ENV_FILE" | cut -d= -f2)
ADMIN_CHAT_ID=$(grep '^ADMIN_CHAT_ID=' "$ENV_FILE" | cut -d= -f2)

if [ -z "$BOT_TOKEN" ] || [ -z "$ADMIN_CHAT_ID" ]; then
    echo "$(date): Missing BOT_TOKEN or ADMIN_CHAT_ID in .env"
    exit 1
fi

ALERT_FILE="/tmp/medexpert_alert_sent"
DEPLOY_FLAG="/tmp/medexpert_deploying"
DEPLOY_NOTIFIED="/tmp/medexpert_deploy_notified"

# If deploying, skip normal checks — just notify once
if [ -f "$DEPLOY_FLAG" ]; then
    if [ ! -f "$DEPLOY_NOTIFIED" ]; then
        curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
            -d chat_id="$ADMIN_CHAT_ID" \
            -d text="MedExpert se esta actualizando, regresa en unos minutos..." \
            -d parse_mode="HTML" > /dev/null
        touch "$DEPLOY_NOTIFIED"
    fi
    exit 0
fi

# If deploy just finished (flag removed but notification was sent), send "back online"
if [ -f "$DEPLOY_NOTIFIED" ]; then
    # Verify bot is actually running before announcing
    if pgrep -f 'python.*bot.py' > /dev/null 2>&1; then
        curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
            -d chat_id="$ADMIN_CHAT_ID" \
            -d text="MedExpert esta de vuelta y funcionando." \
            -d parse_mode="HTML" > /dev/null
        rm -f "$DEPLOY_NOTIFIED" "$ALERT_FILE"
    fi
    exit 0
fi

ISSUES=""

# Check admin (app.py) via HTTP
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 http://localhost:8081/login 2>/dev/null)
if [ "$HTTP_CODE" != "200" ]; then
    ISSUES="${ISSUES}Admin (app.py) NO responde (HTTP ${HTTP_CODE})\n"
fi

# Check bot (bot.py) via process
if ! pgrep -f 'python.*bot.py' > /dev/null 2>&1; then
    ISSUES="${ISSUES}Bot (bot.py) NO esta corriendo\n"
fi

# Check disk space (alert if >90%)
DISK_PCT=$(df -h / | awk 'NR==2 {gsub(/%/,""); print $5}')
if [ "$DISK_PCT" -gt 90 ] 2>/dev/null; then
    ISSUES="${ISSUES}Disco al ${DISK_PCT}%\n"
fi

if [ -n "$ISSUES" ]; then
    # Only alert once per issue (avoid spam every 5 min)
    ISSUES_HASH=$(echo "$ISSUES" | md5 2>/dev/null || echo "$ISSUES" | md5sum | cut -d' ' -f1)
    if [ -f "$ALERT_FILE" ] && [ "$(cat "$ALERT_FILE")" = "$ISSUES_HASH" ]; then
        exit 0  # Same alert already sent
    fi

    MESSAGE="⚠️ MedExpert Alert ($(hostname))

${ISSUES}
$(date '+%Y-%m-%d %H:%M')"

    curl -s -X POST "https://api.telegram.org/bot${BOT_TOKEN}/sendMessage" \
        -d chat_id="$ADMIN_CHAT_ID" \
        -d text="$MESSAGE" \
        -d parse_mode="HTML" > /dev/null

    echo "$ISSUES_HASH" > "$ALERT_FILE"
    echo "$(date): Alert sent — $ISSUES"
else
    # Services OK — clear alert flag so next issue triggers a new alert
    rm -f "$ALERT_FILE"
fi
