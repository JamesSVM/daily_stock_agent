from agents.stock_agent import StockAgent

WATCHLIST = [
    "2330",
    "2317",
    "2382",
    "2454",
    "3017",
    "3661",
    "2303",
    "2603"
]

agent = StockAgent()

results = []

for stock in WATCHLIST:
    result = agent.analyze(stock)

    if result:
        results.append(result)

results = sorted(results, key=lambda x: x["score"], reverse=True)

print("今日波段候選：\n")

for r in results:
    print(
        f"{r['stock_id']}  "
        f"Score: {r['score']}  "
        f"Close: {r['close']}"
    )