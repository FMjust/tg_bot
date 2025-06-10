
def find_fvg_zones(candles, lookback=10):
    """
    Ищет FVG зоны за последние свечи.
    Возвращает список FVG-зон [(low, high), ...]
    """
    fvg_zones = []
    for i in range(2, len(candles)):
        prev = candles[i - 2]
        curr = candles[i]
        # FVG GAP UP (пропуск вниз)
        if prev["low"] > curr["high"]:
            fvg_zones.append((curr["high"], prev["low"]))
        # FVG GAP DOWN (пропуск вверх)
        elif prev["high"] < curr["low"]:
            fvg_zones.append((prev["high"], curr["low"]))
    return fvg_zones

def is_price_in_fvg(price, fvg_zones):
    """
    Проверяет, входит ли цена в хотя бы одну зону FVG
    """
    for low, high in fvg_zones:
        if low <= price <= high:
            return True
    return False
