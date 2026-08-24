from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "database.db"
REPORT_PATH = REPO_ROOT / "reports" / "daily_signal.csv"
ENV_FILE = Path.home() / ".daily_stock_agent.env"


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE settings without overriding explicit shell variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def run_step(label: str, command: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n=== {label} ===")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _send_failure_alert(error: Exception) -> None:
    """Send a lightweight alert without invoking the LLM/report pipeline."""
    try:
        from email_notifier import send_email

        body = (
            "Daily Stock Agent ALERT\n\n"
            "The market-data refresh did not pass its safety checks.\n"
            "No trading signal or BUY_NEXT_OPEN report was generated.\n\n"
            f"Reason: {error}\n\n"
            "Check reports/launchd.log and reports/launchd.error.log.\n"
            "If the failure is Yahoo data freshness, the scheduled retry will try again next run."
        )
        send_email(body, subject="Daily Stock Agent ALERT - Data Refresh Failed")
        print("Alert email: sent")
    except Exception as alert_error:  # noqa: BLE001 - do not hide original refresh failure
        print(f"Alert email: failed ({alert_error})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete daily stock agent")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--period", default="3mo")
    args = parser.parse_args()

    _load_env_file(ENV_FILE)
    env = os.environ.copy()
    env.setdefault("OLLAMA_URL", "http://localhost:11434/api/chat")
    env.setdefault("OLLAMA_MODEL", "qwen3:8b")

    try:
        run_step(
            "Refresh market data",
            [
                sys.executable,
                "scripts/update_daily_prices.py",
                "--db",
                str(DB_PATH),
                "--period",
                args.period,
                "--retry-attempts",
                "3",
                "--retry-wait-seconds",
                "600",
                "--min-coverage",
                "0.90",
            ],
            env,
        )

        command = [sys.executable, "daily_report.py", "--db", str(DB_PATH), "--output", str(REPORT_PATH)]
        if args.send_email:
            command.append("--send-email")

        run_step("Generate signal + LLM explanation + report", command, env)
        print("\nDaily stock agent completed successfully.")
    except subprocess.CalledProcessError as error:
        print(f"\nDaily stock agent aborted safely: step failed with exit code {error.returncode}.")
        _send_failure_alert(error)
        raise
    except Exception as error:
        print(f"\nDaily stock agent aborted safely: {error}")
        _send_failure_alert(error)
        raise


if __name__ == "__main__":
    main()
