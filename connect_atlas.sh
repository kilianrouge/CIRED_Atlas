#!/usr/bin/env bash
# connect_atlas.sh – Open an SSH tunnel to the ATLAS server and start the app.
# Run this script LOCALLY. It will:
#   1. Ask for your username
#   2. Establish an SSH tunnel (local port 5050 → remote port 5050)
#   3. Start the ATLAS app on the server
#   4. Open http://localhost:5050 in your local browser
#
# Usage:  bash connect_atlas.sh [user] [port]

set -e

LOCAL_PORT="${2:-5050}"
REMOTE_PORT="5050"
REMOTE_APP_DIR="/diskdata/cired/rouge/ATLAS"
REMOTE_RUN_CMD="bash ${REMOTE_APP_DIR}/run_atlas.sh ${REMOTE_PORT}"

# ── Collect credentials ───────────────────────────────────────────────────────

HOST="inari.centre-cired.fr"

if [[ -n "$1" ]]; then
    USER_NAME="$1"
else
    read -rp "Username: " USER_NAME
fi

echo ""
echo "  Connecting to ${USER_NAME}@${HOST}"
echo "  Tunnel: localhost:${LOCAL_PORT}  →  remote:${REMOTE_PORT}"
echo "  Command: ${REMOTE_RUN_CMD}"
echo ""

# ── Open browser after a short delay ─────────────────────────────────────────

(
    sleep 3
    URL="http://localhost:${LOCAL_PORT}"
    echo "  Opening ${URL} ..."
    if command -v xdg-open &>/dev/null; then
        xdg-open "$URL"
    elif command -v open &>/dev/null; then
        open "$URL"
    else
        echo "  (could not auto-open browser – navigate to ${URL})"
    fi
) &

# ── SSH with port forwarding – password prompt handled by SSH itself ──────────
# -L  : forward local port to remote port
# -t  : allocate a TTY so the remote process receives Ctrl-C correctly
# Pass-through to SSH for key/password authentication.

ssh -t \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "${USER_NAME}@${HOST}" \
    "${REMOTE_RUN_CMD}"
