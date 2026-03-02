#!/usr/bin/env bash
# run.sh – Start the ATLAS Literature Discovery app
# Usage:  bash run.sh [port]
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default port: 5000 + (UID % 1000) so each Unix user gets their own port
PORT="${1:-$((5000 + $(id -u) % 1000))}"

export ATLAS_PORT="$PORT"

# Kill any previous instance of this app running on the same port
# (left over from a previous Ctrl+C that closed SSH but not the remote process)
PYTHON="/data/software/mambaforge/mambaforge/envs/Shiny_app1_screening_sobriety/bin/python"
OLD_PID=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
if [[ -n "$OLD_PID" ]]; then
    echo "  Killing stale process on port $PORT (PID $OLD_PID)..."
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
fi

cd "$DIR"
# Ensure all files created by this process (cache, DB) are world-writable
# so every user sharing this installation can read and overwrite them.
# Sensitive files (*.key) are explicitly chmod 600'd in code, overriding this.
umask 000
exec "$PYTHON" app.py
