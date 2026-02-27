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

# 4. Sync DB and restart bot
echo ""
echo "[4/4] Syncing DB and restarting bot..."
scp -q "$LOCAL_PATH/data/medexpert_admin.db" "$M1_HOST:$M1_PATH/data/medexpert_admin.db"

ssh "$M1_HOST" "
    pkill -f 'python bot.py' 2>/dev/null || true
    sleep 1
    cd $M1_PATH && source venv/bin/activate
    nohup python bot.py --specialty oncologia > /tmp/bot.log 2>&1 &
    sleep 2
    cat /tmp/bot.log
"

echo ""
echo "=== Deploy complete ==="
