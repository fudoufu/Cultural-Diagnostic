#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./refresh_app.sh            # uses port 8501
#   ./refresh_app.sh 8502       # uses custom port

PORT="${1:-8501}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Repo: ${REPO_DIR}"
cd "${REPO_DIR}"

echo "==> Syncing with origin/main"
git fetch origin
git checkout main
git reset --hard origin/main

echo "==> Latest commit"
git log --oneline -1

if [[ ! -d ".venv" ]]; then
  echo "==> Creating virtual environment (.venv)"
  python3 -m venv .venv
fi

echo "==> Activating virtual environment"
source .venv/bin/activate

echo "==> Installing dependencies"
python -m pip install -r requirements.txt

echo "==> Restarting Streamlit (port ${PORT})"
pkill -f "streamlit run app.py" || true

echo "==> Starting app"
exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port "${PORT}"
