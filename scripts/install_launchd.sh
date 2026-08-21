#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLIST_LABEL="com.jamessvm.daily-stock-agent"
PLIST_PATH="$HOME/Library/LaunchAgents/${PLIST_LABEL}.plist"
RUNNER="$REPO_ROOT/scripts/run_scheduled_agent.sh"
LOG_DIR="$REPO_ROOT/reports"
UID_VALUE="$(id -u)"

if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  echo "Missing virtualenv Python at $REPO_ROOT/.venv/bin/python"
  echo "Create/restore the project .venv first."
  exit 1
fi

if [ ! -f "$HOME/.daily_stock_agent.env" ]; then
  echo "Missing $HOME/.daily_stock_agent.env"
  echo "Create it with the SMTP and Ollama settings before installing launchd."
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
chmod 600 "$HOME/.daily_stock_agent.env"
chmod +x "$RUNNER"

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${PLIST_LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${RUNNER}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${REPO_ROOT}</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>OLLAMA_URL</key>
    <string>http://localhost:11434/api/chat</string>
    <key>OLLAMA_MODEL</key>
    <string>qwen3:8b</string>
    <key>OLLAMA_TIMEOUT_SECONDS</key>
    <string>120</string>
  </dict>

  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
    <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>18</integer><key>Minute</key><integer>30</integer></dict>
  </array>

  <key>StandardOutPath</key>
  <string>${LOG_DIR}/launchd.log</string>
  <key>StandardErrorPath</key>
  <string>${LOG_DIR}/launchd.error.log</string>

  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
EOF

# Replace an existing job safely, then bootstrap the new plist.
launchctl bootout "gui/${UID_VALUE}" "$PLIST_PATH" 2>/dev/null || true
launchctl bootstrap "gui/${UID_VALUE}" "$PLIST_PATH"
launchctl enable "gui/${UID_VALUE}/${PLIST_LABEL}" || true

echo "Installed: $PLIST_PATH"
echo "Schedule: Monday-Friday at 18:30 (local Mac time)"
echo "Model: qwen3:8b"
echo "Logs: $LOG_DIR/launchd.log and $LOG_DIR/launchd.error.log"
launchctl print "gui/${UID_VALUE}/${PLIST_LABEL}"
