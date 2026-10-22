#!/usr/bin/env bash
# The OA-RLAIF agent console (Claude-Code style), everything in this folder:
#   ./oa_agent.sh       banner (model, where it is loaded from, thinking, skills), then it asks for the two ontologies
#                       and the output folder, and asks before it starts; the agent asks before every command
set -euo pipefail
. "$(cd "$(dirname "$0")" && pwd)/setup_ollama.sh"
exec "$RLAIF/.venv/bin/python" "$HERE/oa_agent.py" "$@"
