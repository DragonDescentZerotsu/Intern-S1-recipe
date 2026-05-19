#!/usr/bin/env bash
# Start a local HTTP server for viewer.html.
#
# Usage:
#   bash reasoning-trajectory/start_viewer.sh [port]
#
# Then open:
#   http://localhost:8765/viewer.html
#
# The page auto-discovers run folders and task JSONL files from this directory.
# You can also upload JSONL files manually, or enter a server-relative path
# such as:
#   test_gpt-oss-20b-base-no-tool_20260428_201656.log/DILI.jsonl
#
# If your browser runs on a different machine, use SSH port forwarding:
#   ssh -L 8765:127.0.0.1:8765 <user>@<server>
# and open http://localhost:8765/viewer.html on your local computer.

set -euo pipefail

cd "$(dirname "$0")"
PORT="${1:-8777}"

echo "==============================================="
echo " TDC Reasoning Trace Viewer"
echo " Directory: $(pwd)"
echo " Port:      $PORT"
echo ""
echo " Open:"
echo "   http://localhost:$PORT/viewer.html"
echo ""
echo " Remote browser access:"
echo "   ssh -L $PORT:127.0.0.1:$PORT <user>@<server>"
echo "   then open http://localhost:$PORT/viewer.html locally"
echo ""
echo " JSONL path examples inside the page:"
echo "   Prefer the run/task dropdowns; manual paths also work."
echo "   test_gpt-oss-20b-base-no-tool_20260428_201656.log/DILI.jsonl"
echo "   reasoning-trajectory/test_gpt-oss-20b-base-no-tool_20260428_201656.log/DILI.jsonl"
echo "   /data2/tianang/projects/Intern-S1/reasoning-trajectory/test_gpt-oss-20b-base-no-tool_20260428_201656.log/DILI.jsonl"
echo ""
echo " Press Ctrl+C to stop the server."
echo "==============================================="

if command -v python3 >/dev/null 2>&1; then
  python3 -m http.server "$PORT"
elif command -v python >/dev/null 2>&1; then
  python -m SimpleHTTPServer "$PORT"
else
  echo "Error: python not found. Run manually from this folder: python3 -m http.server $PORT"
  exit 1
fi
