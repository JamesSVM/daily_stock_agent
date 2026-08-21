# Local V1.5 smoke test

This test runs the real local pipeline against the SQLite database and the local Ollama server.
It does **not** send email.

## Prerequisites

- Ollama server running locally
- The configured Ollama model available locally
- `data/database.db` present on your machine
- Python dependencies installed from `requirements.txt`

## Run

From the repository root:

```bash
python scripts/smoke_test_local.py
```

Optional environment variables:

```bash
export OLLAMA_URL=http://localhost:11434/api/chat
export OLLAMA_MODEL=qwen3:8b
export OLLAMA_TIMEOUT_SECONDS=120
export STOCK_DB_PATH=data/database.db
export DAILY_REPORT_OUTPUT=reports/daily_signal.csv
python scripts/smoke_test_local.py
```

## Expected behavior

The script should:

1. Load the latest completed trading date from SQLite.
2. Generate deterministic V1.5 candidates.
3. Send only selected signal facts to Ollama for explanation.
4. Produce a plain-text report under `reports/`.
5. Print `Smoke test completed. Email was not sent.`

The LLM must not change `BUY_NEXT_OPEN` / `WATCH`. The strategy engine remains authoritative.

## Email test

Do not add SMTP credentials to source control. Once the local smoke test succeeds, configure the required SMTP environment variables and run:

```bash
python daily_report.py --send-email
```

The email step is deliberately separate from the smoke test so a model/runtime test cannot accidentally send a real notification.
