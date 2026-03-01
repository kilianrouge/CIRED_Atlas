#!/usr/bin/env bash
# run.sh – Start the ATLAS Literature Discovery app
# Usage:  bash run.sh [port]
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-5050}"

echo ""
echo "  ╔══════════════════════════════════════════════╗"
echo "  ║  ATLAS - Literature Discovery                ║"
echo "  ║  Starting on http://127.0.0.1:${PORT}          ║"
echo "  ╚══════════════════════════════════════════════╝"
echo ""

export ATLAS_PORT="$PORT"

PYTHON="/data/software/mambaforge/mambaforge/envs/Shiny_app1_screening_sobriety/bin/python"

cd "$DIR"
exec "$PYTHON" app.py
