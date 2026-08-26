from __future__ import annotations

"""Build the live Taiwan-stock universe from official TWSE/TPEx snapshots.

The live strategy intentionally scans common stocks only. ETFs, ETNs, warrants,
and other non-common-stock instruments are excluded. The remaining universe is
ranked by the latest reported transaction amount and capped at TARGET_COUNT,
which keeps the scan liquid without making a subjective fundamental judgement.
"""

import argparse
import csv
import json
import sqlite3
import ssl
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import certifi

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "database.db"
REPORT_PATH = REPO_ROOT / "reports" / "stock_universe.csv"
TAIPEI_TZ = ZoneInfo("Asia/Taipei")

TWSE_URL = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
TPEX_URL = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
TARGET_COUNT = 300

# macOS Homebrew Python 3.14 can reject otherwise valid public certificates
# when its default trust store is incomplete. certifi provides a portable CA
# bundle without disabling TLS verification.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _fetch_json(url: str) -> list[dict]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "daily-stock-agent/1.6"},
    )
    with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected API response from {url}")
    return [row for row in payload if isinstance(row, dict)]


def _clean_number(value: object) -> float:
    text = str(value or "").replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        return 0.0


def _is_common_stock_code(code: object) -> bool:
    code = str(code or "").strip().upper()
    # Ordinary Taiwan listed/OTC shares use four numeric digits. ETF/ETN codes
    # commonly begin with 00, while warrants and other derivatives are longer
    # or contain letters. This deliberately excludes them from the live universe.
    return len(code) == 4 and code.isdigit() and not code.startswith("00")


def _normalize_twse(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        code = str(row.get("Code", "")).strip()
        if not _is_common_stock_code(code):
            continue
        result.append(
            {
                "stock_id": code,
                "stock_name": str(row.get("Name", "")).strip(),
                "market": "TWSE",
                "transaction_amount": _clean_number(row.get("TradeValue")),
            }
        )
    return result


def _normalize_tpex(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        code = str(row.get("SecuritiesCompanyCode", "")).strip()
        if not _is_common_stock_code(code):
            continue
        result.append(
            {
                "stock_id": code,
                "stock_name": str(row.get("CompanyName", "")).strip(),
                "market": "TPEX",
                "transaction_amount": _clean_number(row.get("TransactionAmount")),
            }
        )
    return result


def build_universe(rows: list[dict], target_count: int = TARGET_COUNT) -> list[dict]:
    # If a code ever appears in both feeds, keep the record with the larger
    # transaction amount so the selection remains deterministic.
    by_code: dict[str, dict] = {}
    for row in rows:
        code = row["stock_id"]
        current = by_code.get(code)
        if current is None or row["transaction_amount"] > current["transaction_amount"]:
            by_code[code] = row

    ranked = sorted(
        by_code.values(),
        key=lambda row: (-row["transaction_amount"], row["stock_id"]),
    )
    return ranked[:target_count]


def refresh_universe(db_path: str, target_count: int = TARGET_COUNT) -> int:
    twse = _normalize_twse(_fetch_json(TWSE_URL))
    tpex = _normalize_tpex(_fetch_json(TPEX_URL))
    universe = build_universe(twse + tpex, target_count=target_count)

    if len(universe) < target_count:
        raise RuntimeError(
            f"Only {len(universe)} eligible common stocks were returned; "
            f"target is {target_count}. No universe update applied."
        )

    updated_at = datetime.now(TAIPEI_TZ).isoformat()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stock_universe (
                stock_id TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                market TEXT NOT NULL,
                transaction_amount REAL NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("UPDATE stock_universe SET is_active = 0")
        conn.executemany(
            """
            INSERT INTO stock_universe
                (stock_id, stock_name, market, transaction_amount, is_active, updated_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(stock_id) DO UPDATE SET
                stock_name=excluded.stock_name,
                market=excluded.market,
                transaction_amount=excluded.transaction_amount,
                is_active=1,
                updated_at=excluded.updated_at
            """,
            [
                (
                    row["stock_id"],
                    row["stock_name"],
                    row["market"],
                    row["transaction_amount"],
                    updated_at,
                )
                for row in universe
            ],
        )
        conn.commit()

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank", "stock_id", "stock_name", "market", "transaction_amount"],
        )
        writer.writeheader()
        for rank, row in enumerate(universe, start=1):
            writer.writerow({"rank": rank, **row})

    twse_count = sum(row["market"] == "TWSE" for row in universe)
    tpex_count = sum(row["market"] == "TPEX" for row in universe)
    print(f"Universe target: {target_count}")
    print(f"Universe selected: {len(universe)}")
    print(f"TWSE: {twse_count} | TPEX: {tpex_count}")
    print(f"Saved: {REPORT_PATH}")
    return len(universe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the 300-stock live universe")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--target-count", type=int, default=TARGET_COUNT)
    args = parser.parse_args()
    refresh_universe(args.db, target_count=args.target_count)


if __name__ == "__main__":
    main()
