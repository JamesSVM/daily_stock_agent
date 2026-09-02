from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = Path.home() / ".daily_stock_agent.env"
DB_PATH = REPO_ROOT / "data" / "database.db"
REPORT_PATH = REPO_ROOT / "reports" / "daily_signal.txt"
PORTFOLIO_STATE_PATH = REPO_ROOT / "data" / "portfolio_state.json"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def run_step(name: str, command: list[str], env: dict[str, str]) -> None:
    print(f"\n=== {name} ===")
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def _send_failure_alert(error: Exception) -> None:
    try:
        command = [sys.executable, "email_notifier.py", "--failure", str(error)]
        subprocess.run(command, cwd=REPO_ROOT, env=os.environ.copy(), check=False)
    except Exception as alert_error:  # noqa: BLE001 - alert must never hide root cause
        print(f"Alert email: failed ({alert_error})")


def _run_performance_tracking(env: dict[str, str]) -> None:
    print("\n=== Build V1.6 performance tracking ===")
    try:
        run_step(
            "Build V1.6 performance tracking",
            [
                sys.executable,
                "scripts/build_performance_tracking.py",
                "--db",
                str(DB_PATH),
            ],
            env,
        )
    except Exception as error:  # noqa: BLE001 - analytics must not block live delivery
        print(f"V1.6 performance tracking warning: {error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete daily stock agent")
    parser.add_argument("--send-email", action="store_true")
    parser.add_argument("--period", default="3mo")
    parser.add_argument(
        "--portfolio-state",
        default=str(PORTFOLIO_STATE_PATH),
        help="JSON file containing capital, positions and trade history",
    )
    args = parser.parse_args()

    _load_env_file(ENV_FILE)
    env = os.environ.copy()
    env.setdefault("OLLAMA_URL", "http://localhost:11434/api/chat")
    env.setdefault("OLLAMA_MODEL", "qwen3:8b")

    try:
        run_step(
            "Refresh stock universe (300 common stocks)",
            [
                sys.executable,
                "scripts/refresh_stock_universe.py",
                "--db",
                str(DB_PATH),
                "--target-count",
                "300",
            ],
            env,
        )

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
                "10",
                "--min-coverage",
                "0.90",
            ],
            env,
        )

        command = [
            sys.executable,
            "daily_report.py",
            "--db",
            str(DB_PATH),
            "--output",
            str(REPORT_PATH),
            "--portfolio-state",
            args.portfolio_state,
        ]
        if args.send_email:
            command.append("--send-email")

        run_step("Generate V1.6 signal + portfolio decision + report", command, env)
        _run_performance_tracking(env)
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
