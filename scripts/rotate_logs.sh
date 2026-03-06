#!/bin/bash
# Log rotation for MedExpert services
# Cron: 0 4 * * * /Users/juanma/Projects/medexpert-admin/scripts/rotate_logs.sh

LOG_DIR="/tmp"
ARCHIVE_DIR="/Users/juanma/logs/medexpert"
DAYS_TO_KEEP=7
MAX_SIZE_MB=50

mkdir -p "$ARCHIVE_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

for LOG_NAME in admin bot; do
    LOG_FILE="$LOG_DIR/${LOG_NAME}.log"

    [ -f "$LOG_FILE" ] || continue

    SIZE_KB=$(du -k "$LOG_FILE" | cut -f1)

    # Skip if less than 1MB
    if [ "$SIZE_KB" -lt 1024 ]; then
        continue
    fi

    # Archive: copy, compress, truncate (keeps file handle for launchd processes)
    cp "$LOG_FILE" "$ARCHIVE_DIR/${LOG_NAME}_${TIMESTAMP}.log"
    gzip "$ARCHIVE_DIR/${LOG_NAME}_${TIMESTAMP}.log"
    : > "$LOG_FILE"

    echo "$(date): Rotated ${LOG_NAME}.log (${SIZE_KB}KB) -> ${LOG_NAME}_${TIMESTAMP}.log.gz"
done

# Remove archives older than N days
find "$ARCHIVE_DIR" -name "*.log.gz" -mtime +${DAYS_TO_KEEP} -delete
