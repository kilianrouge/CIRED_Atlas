#!/usr/bin/env bash
# connect_atlas.sh – Open an SSH tunnel to the ATLAS server and start the app.
# Run this script LOCALLY. It will:
#   1. Ask for your username (once – all SSH calls share one connection)
#   2. Resolve your personal remote port (5000 + UID % 1000) on the server
#   3. Start your ATLAS instance on the server (if not already running)
#   4. Establish an SSH tunnel  local_port → remote_port
#   5. Open http://localhost:<local_port> in your local browser
#   6. Keep the tunnel alive until you press Ctrl-C (then clean up)
#
# Usage:  bash connect_atlas.sh [user] [local_port]

set -e

LOCAL_PORT="${2:-5050}"
REMOTE_APP_DIR="/data/shared/ATLAS"
HOST="inari.centre-cired.fr"

# ── Collect username ──────────────────────────────────────────────────────────
if [[ -n "$1" ]]; then
    USER_NAME="$1"
else
    read -rp "Username: " USER_NAME
fi

# ── SSH ControlMaster: one auth, all connections share it ────────────────────
CTRL="/tmp/atlas_ssh_${USER_NAME}"
cleanup() {
    echo ""
    echo "  Closing tunnel…"
    ssh -O stop -S "$CTRL" "${USER_NAME}@${HOST}" 2>/dev/null || true
    # Kill local port-forward process if still around
    [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "  Connecting to ${USER_NAME}@${HOST}…"
# Open the master connection (asks for password/key exactly once)
ssh -fNM -S "$CTRL" -o ControlPersist=60 \
    -o StrictHostKeyChecking=accept-new \
    "${USER_NAME}@${HOST}"

# ── Resolve per-user remote port ─────────────────────────────────────────────
REMOTE_PORT=$(ssh -S "$CTRL" "${USER_NAME}@${HOST}" 'echo $((5000 + $(id -u) % 1000))')
echo "  Tunnel:  localhost:${LOCAL_PORT}  →  ${HOST}:${REMOTE_PORT}"

# ── Kill any stale local listener on LOCAL_PORT ───────────────────────────────
OLD_TUNNEL=$(lsof -ti tcp:"${LOCAL_PORT}" 2>/dev/null || true)
if [[ -n "$OLD_TUNNEL" ]]; then
    echo "  Clearing stale process on local port ${LOCAL_PORT}…"
    kill "$OLD_TUNNEL" 2>/dev/null || true
    sleep 1
fi

# ── Start the app on the remote host (idempotent – run_atlas.sh kills stale) ──
LOG="/tmp/atlas_${USER_NAME}.log"
echo "  Starting remote app (log: ${HOST}:${LOG})"
ssh -S "$CTRL" "${USER_NAME}@${HOST}" \
    "nohup bash ${REMOTE_APP_DIR}/run_atlas.sh ${REMOTE_PORT} </dev/null >>\"${LOG}\" 2>&1 & disown; sleep 2"

# ── Open the SSH tunnel in the background ────────────────────────────────────
ssh -S "$CTRL" -fNL "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${USER_NAME}@${HOST}"
TUNNEL_PID=$(lsof -ti tcp:"${LOCAL_PORT}" 2>/dev/null || true)

# ── Wait for the app to respond, then open the browser ───────────────────────
URL="http://localhost:${LOCAL_PORT}"
echo "  Waiting for app to be ready…"
for i in $(seq 1 30); do
    sleep 2
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 "${URL}" 2>/dev/null || true)
    if [[ "$HTTP_CODE" == "200" ]]; then
        echo "  ✅ Ready → ${URL}"
        echo ""
        # Open browser: macOS → open, Linux → xdg-open, fallback print URL
        if command -v open &>/dev/null; then
            open "${URL}"
        elif command -v xdg-open &>/dev/null; then
            xdg-open "${URL}"
        fi
        break
    fi
done

echo "  Tunnel active. Press Ctrl-C to disconnect, or it will close automatically when the app stops."
# Poll the app every 10 s; exit (triggering cleanup via trap) if it stops responding
while true; do
    sleep 10
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "${URL}" 2>/dev/null || true)
    if [[ "$HTTP_CODE" != "200" ]]; then
        echo ""
        echo "  ⚠️  App no longer reachable (HTTP ${HTTP_CODE}). Closing tunnel…"
        exit 1
    fi
done
