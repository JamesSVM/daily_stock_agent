from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "database.db"
REPORT_PATH = REPO_ROOT / "reports" / "daily_signal.csv"


def run_step(label: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n=== {label} ===")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete daily stock agent")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--period", default="3mo")
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("OLLAMA_URL", "http://localhost:11434/api/chat")
    env.setdefault("OLLAMA_MODEL", "qwen2.5-coder:7b")

    run_step(
        "Refresh market data",
        [sys.executable, "scripts/update_daily_prices.py", "--db", str(DB_PATH), "--period", args.period],
        env,
    )

    command = [
        sys.executable,
        "daily_report.py",
        "--db",
        str(DB_PATH),
        "--output",
        str(REPORT_PATH),
    ]
    if args.send_email:
        command.append("--send-email")

    run_step("Generate signal + LLM explanation + report", command, env)
    print("\nDaily stock agent completed successfully.")


if __name__ == "__main__":
    main()
