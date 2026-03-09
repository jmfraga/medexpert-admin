#!/bin/bash
# Deploy medexpert-admin to Mac Mini M1
# Usage: ./deploy.sh [--no-push] [--no-chromadb]
#
# Services are managed by launchd (KeepAlive):
#   com.medexpert.admin  — app.py --port 8081
#   com.medexpert.bot    — bot.py --specialty oncologia
# Restart via: launchctl kickstart -k gui/$(id -u)/<label>

set -e

M1_HOST="juanma@100.107.30.22"
M1_PATH="~/Projects/medexpert-admin"
LOCAL_PATH="$(cd "$(dirname "$0")" && pwd)"

NO_PUSH=false
NO_CHROMADB=false

for arg in "$@"; do
    case $arg in
        --no-push) NO_PUSH=true ;;
        --no-chromadb) NO_CHROMADB=true ;;
    esac
done

echo "=== MedExpert Deploy ==="

# 1. Push to repo
if [ "$NO_PUSH" = false ]; then
    echo ""
    echo "[1/4] Pushing to GitHub..."
    git -C "$LOCAL_PATH" push
else
    echo ""
    echo "[1/4] Skipping git push (--no-push)"
fi

# 2. Sync code to M1
echo ""
echo "[2/4] Syncing code to M1..."
rsync -avz --delete \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '.env' \
    --exclude 'data/' \
    --exclude '.DS_Store' \
    "$LOCAL_PATH/" "$M1_HOST:$M1_PATH/" | tail -3

# 3. Sync ChromaDB if requested
if [ "$NO_CHROMADB" = false ]; then
    echo ""
    echo "[3/4] Syncing ChromaDB oncologia..."
    rsync -avz \
        "$LOCAL_PATH/data/experts/oncologia/chromadb/" \
        "$M1_HOST:$M1_PATH/data/experts/oncologia/chromadb/" | tail -3
else
    echo ""
    echo "[3/4] Skipping ChromaDB sync (--no-chromadb)"
fi

# 4. Restart services via launchctl (DB stays on M1 — production data is authoritative)
echo ""
echo "[4/4] Restarting services via launchctl..."

# Clean Python cache to force fresh bytecode
ssh "$M1_HOST" "rm -rf $M1_PATH/__pycache__" 2>/dev/null || true

# Restart admin (launchctl kickstart -k = kill + relaunch, instant)
echo "Restarting admin..."
ssh "$M1_HOST" "launchctl kickstart -k gui/\$(id -u)/com.medexpert.admin"
sleep 3
ssh "$M1_HOST" "tail -3 /tmp/admin.log"

# Create deploy flag (health check will skip alerts, bot shows maintenance message)
ssh "$M1_HOST" "echo \$(date +%s) > /tmp/medexpert_deploying"

# Restart bot (unload → wait for Telegram to release polling → load)
echo "Stopping bot..."
ssh "$M1_HOST" "launchctl unload ~/Library/LaunchAgents/com.medexpert.bot.plist 2>/dev/null; sleep 1; pkill -9 -f 'bot.py' 2>/dev/null || true"
echo "Waiting 60s for Telegram to release polling lock..."
sleep 60
echo "Starting bot..."
ssh "$M1_HOST" "> /tmp/bot.log; launchctl load ~/Library/LaunchAgents/com.medexpert.bot.plist"
sleep 12
ssh "$M1_HOST" "tail -8 /tmp/bot.log"

# Remove deploy flag
ssh "$M1_HOST" "rm -f /tmp/medexpert_deploying"

# Verify both running
echo ""
echo "Verifying services..."
ssh "$M1_HOST" "launchctl list | grep com.medexpert"

echo ""
echo "=== Deploy complete ==="
