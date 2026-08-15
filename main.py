from crawler.price import get_price
from engine.feature_engine import build_features
from engine.scoring_engine import calculate_score
from engine.backtest_engine import run_backtest
from engine.metrics import evaluate

df = get_price("2382", period="2y")

df = build_features(df)

trades = run_backtest(
    df,
    calculate_score
)

print(trades.head())

print(evaluate(trades))