# engine/scoring_engine.py

def calculate_score(df):
    """向量化計算分數，加入乖離率剔除與高標準爆發力"""
    score = df["Close"] * 0

    # 1. 爆發力特徵 (門檻拉高)
    score += (df["Close"] > df["ma5"]).astype(int) * 20
    score += (df["ma5"] > df["ma10"]).astype(int) * 10   # 均線必須完美多頭排列
    score += (df["ma10"] > df["ma20"]).astype(int) * 10
    score += (df["ma20"] > df["ma60"]).astype(int) * 20
    score += (df["volume_ratio"] > 2.0).astype(int) * 20 # 嚴格要求 2 倍以上爆量
    score += (df["breakout_ratio"] > 0.98).astype(int) * 20

    # 2. 乖離率剔除 (風險控制濾網)
    # 如果收盤價距離 20 日線已經超過 15%，代表已經漲了一波，追高風險極大
    is_over_extended = df["bias_ma20"] > 0.15

    # 風控 B: 法人籌碼背書 (近 3 日投信累計買超 > 0 或外資近 3 日大買 > 1000 張)
    has_institutional_support = (df["Trust_Buy_3D"] > 0) | (df["Foreign_Buy_3D"] > 1000)

    # 3. 複合一票否決 (只要符合任一違規條件，立即歸零)
    # 違規情況：【乖離率過大】 或 【缺乏法人籌碼支持】
    veto_condition = is_over_extended | (~has_institutional_support)

    score = score.mask(veto_condition, 0)

    return score