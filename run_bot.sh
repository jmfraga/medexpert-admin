#!/bin/bash
# Run bot with auto-restart on crash/conflict
cd "$(dirname "$0")"
source venv/bin/activate

while true; do
    echo "$(date): Starting bot..."
    python bot.py --specialty oncologia 2>&1 | tee -a /tmp/bot.log
    EXIT_CODE=$?
    echo "$(date): Bot exited (code $EXIT_CODE), restarting in 15s..."
    sleep 15
done
