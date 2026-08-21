#!/bin/bash
set -euo pipefail

REPO_ROOT="/Users/USERNAME/daily_stock_agent"
ENV_FILE="$HOME/.daily_stock_agent.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing $ENV_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

cd "$REPO_ROOT"
exec "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/scripts/run_daily_agent.py" --send-email
