#!/bin/bash
# Daily SQLite backup with 7-day rotation
# Cron: 0 3 * * * /Users/juanma/Projects/medexpert-admin/scripts/backup_db.sh

set -e

DB_PATH="/Users/juanma/Projects/medexpert-admin/data/medexpert_admin.db"
BACKUP_DIR="/Users/juanma/backups/medexpert"
DAYS_TO_KEEP=7

mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/medexpert_admin_${TIMESTAMP}.db"

# Use sqlite3 .backup for a safe, consistent copy (handles WAL mode)
sqlite3 "$DB_PATH" ".backup '$BACKUP_FILE'"

# Compress
gzip "$BACKUP_FILE"

# Remove backups older than N days
find "$BACKUP_DIR" -name "medexpert_admin_*.db.gz" -mtime +${DAYS_TO_KEEP} -delete

# Log
echo "$(date): Backup OK -> ${BACKUP_FILE}.gz ($(du -h "${BACKUP_FILE}.gz" | cut -f1))"
