
def find_swing_highs_lows(candles, lookback=3):
    """
    Находит swing high/low на основе соседних свечей.
    Возвращает 2 последних high и low [(highs), (lows)]
    """
    highs = []
    lows = []
    for i in range(lookback, len(candles) - lookback):
        center = candles[i]
        neighbors = candles[i - lookback:i] + candles[i + 1:i + 1 + lookback]

        if all(center["high"] > c["high"] for c in neighbors):
            highs.append(center["high"])
        if all(center["low"] < c["low"] for c in neighbors):
            lows.append(center["low"])

    return highs[-2:], lows[-2:]

def is_near_liquidity(price, levels, tolerance=0.01):
    """
    Проверяет, находится ли цена вблизи хотя бы одного уровня.
    """
    for level in levels:
        if abs(price - level) / level <= tolerance:
            return True
    return False
