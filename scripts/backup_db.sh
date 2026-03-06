#!/bin/bash
# Daily full backup of MedExpert production data
# Cron: 0 3 * * * /Users/juanma/Projects/medexpert-admin/scripts/backup_db.sh

set -e

PROJECT="/Users/juanma/Projects/medexpert-admin"
BACKUP_DIR="/Users/juanma/backups/medexpert"
DAYS_TO_KEEP=7

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. SQLite — safe backup (handles WAL mode)
DB_BACKUP="$BACKUP_DIR/medexpert_admin_${TIMESTAMP}.db"
sqlite3 "$PROJECT/data/medexpert_admin.db" ".backup '$DB_BACKUP'"
gzip "$DB_BACKUP"
echo "$(date): DB backup -> ${DB_BACKUP}.gz ($(du -h "${DB_BACKUP}.gz" | cut -f1))"

# 2. .env (production secrets — small but critical)
cp "$PROJECT/.env" "$BACKUP_DIR/env_${TIMESTAMP}.bak"
gzip "$BACKUP_DIR/env_${TIMESTAMP}.bak"
echo "$(date): .env backup OK"

# 3. Verification documents (user uploads)
if [ -d "$PROJECT/data/verifications" ]; then
    tar czf "$BACKUP_DIR/verifications_${TIMESTAMP}.tar.gz" -C "$PROJECT/data" verifications/
    echo "$(date): Verifications backup -> $(du -h "$BACKUP_DIR/verifications_${TIMESTAMP}.tar.gz" | cut -f1)"
fi

# 4. ChromaDB (711MB — weekly only, on Sundays)
DOW=$(date +%u)  # 1=Monday, 7=Sunday
if [ "$DOW" -eq 7 ]; then
    tar czf "$BACKUP_DIR/chromadb_${TIMESTAMP}.tar.gz" -C "$PROJECT/data/experts/oncologia" chromadb/
    echo "$(date): ChromaDB backup -> $(du -h "$BACKUP_DIR/chromadb_${TIMESTAMP}.tar.gz" | cut -f1)"
    # ChromaDB backups: keep 4 weeks
    find "$BACKUP_DIR" -name "chromadb_*.tar.gz" -mtime +28 -delete
fi

# Cleanup old daily backups
find "$BACKUP_DIR" -name "medexpert_admin_*.db.gz" -mtime +${DAYS_TO_KEEP} -delete
find "$BACKUP_DIR" -name "env_*.bak.gz" -mtime +${DAYS_TO_KEEP} -delete
find "$BACKUP_DIR" -name "verifications_*.tar.gz" -mtime +${DAYS_TO_KEEP} -delete

echo "$(date): Backup complete"
