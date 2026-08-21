from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from daily_report import run


def main() -> int:
    db_path = os.getenv("STOCK_DB_PATH", "data/database.db")
    output_path = os.getenv("DAILY_REPORT_OUTPUT", "reports/daily_signal.csv")

    print("Running local V1.5 smoke test")
    print(f"Database: {db_path}")
    print(f"Ollama: {os.getenv('OLLAMA_URL', 'http://localhost:11434/api/chat')}")
    print(f"Model: {os.getenv('OLLAMA_MODEL', 'qwen3:8b')}")
    print("Email: disabled")

    run(db_path=db_path, output_path=output_path, send_notification=False)
    print("Smoke test completed. Email was not sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
