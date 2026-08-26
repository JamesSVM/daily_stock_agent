from scripts.refresh_stock_universe import _is_common_stock_code, build_universe


def test_common_stock_filter_excludes_etf_and_warrant_codes():
    assert _is_common_stock_code("2330")
    assert not _is_common_stock_code("0050")
    assert not _is_common_stock_code("00631L")
    assert not _is_common_stock_code("123456")
    assert not _is_common_stock_code("2330.TW")


def test_build_universe_ranks_by_transaction_amount_and_caps_count():
    rows = [
        {
            "stock_id": "2330",
            "stock_name": "台積電",
            "market": "TWSE",
            "transaction_amount": 300,
        },
        {
            "stock_id": "2317",
            "stock_name": "鴻海",
            "market": "TWSE",
            "transaction_amount": 500,
        },
        {
            "stock_id": "2454",
            "stock_name": "聯發科",
            "market": "TWSE",
            "transaction_amount": 400,
        },
    ]

    universe = build_universe(rows, target_count=2)
    assert [row["stock_id"] for row in universe] == ["2317", "2454"]
