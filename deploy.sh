#!/bin/bash
# Deploy medexpert-admin to Mac Mini M1
# Usage: ./deploy.sh [--no-push] [--no-chromadb]

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

# 4. Restart services (DB stays on M1 — production data is authoritative)
echo ""
echo "[4/4] Restarting services..."

# Restart admin web server
echo "Restarting admin..."
ssh "$M1_HOST" "pkill -9 -f 'python app.py' 2>/dev/null; sleep 2; cd $M1_PATH && nohup ./venv/bin/python app.py --port 8081 > /tmp/admin.log 2>&1 &"
sleep 3
ssh "$M1_HOST" "ps aux | grep 'python app.py' | grep -v grep | awk '{print \"  Admin PID:\", \$2}' || echo '  WARNING: admin not running'"

# Clean Python cache to force fresh bytecode
ssh "$M1_HOST" "rm -rf $M1_PATH/__pycache__" 2>/dev/null || true

echo "Stopping bot..."
# Kill wrapper (run_bot.sh) and all bot instances
ssh "$M1_HOST" "pkill -9 -f 'run_bot.sh' 2>/dev/null; pkill -9 -f 'python bot.py' 2>/dev/null; sleep 2; ps aux | grep -E 'run_bot|python bot' | grep -v grep || echo '  All stopped'"
echo "Starting bot (with auto-restart wrapper)..."
ssh "$M1_HOST" "cd $M1_PATH && nohup bash run_bot.sh > /dev/null 2>&1 &"
sleep 20
ssh "$M1_HOST" "tail -10 /tmp/bot.log"
# Verify running
ssh "$M1_HOST" "ps aux | grep 'python bot.py' | grep -v grep | awk '{print \"  PID:\", \$2}' || echo '  WARNING: bot not running yet (may be retrying)'"

echo ""
echo "=== Deploy complete ==="
