from __future__ import annotations

import argparse
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore a quarantined stock ticker")
    parser.add_argument("stock_id")
    parser.add_argument("--db", default="data/database.db")
    args = parser.parse_args()

    with sqlite3.connect(args.db) as conn:
        conn.execute(
            """
            UPDATE price_update_status
            SET quarantined=0, consecutive_failures=0, last_error=NULL
            WHERE stock_id=?
            """,
            (str(args.stock_id).strip(),),
        )
        conn.commit()

    print(f"Restored stock {args.stock_id}; it will be fetched on the next refresh.")


if __name__ == "__main__":
    main()
