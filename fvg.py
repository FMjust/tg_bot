def find_fvg_zones(candles, lookback=10):
    """
    Ищет FVG зоны за последние свечи.
    Возвращает список FVG-зон в виде словарей:
    [{'type': 'fvg', 'direction': 'long'/'short', 'from': low, 'to': high}, ...]
    """
    fvg_zones = []
    for i in range(2, len(candles)):
        prev = candles[i - 2]
        curr = candles[i]
        # FVG GAP UP (пропуск вниз)
        if prev["low"] > curr["high"]:
            fvg_zones.append({
                'type': 'fvg',
                'direction': 'short',
                'from': curr["high"],
                'to': prev["low"]
            })
        # FVG GAP DOWN (пропуск вверх)
        elif prev["high"] < curr["low"]:
            fvg_zones.append({
                'type': 'fvg',
                'direction': 'long',
                'from': prev["high"],
                'to': curr["low"]
            })
    return fvg_zones

def is_price_in_fvg(price, fvg_zones):
    """
    Проверяет, входит ли цена в хотя бы одну зону FVG
    """
    for zone in fvg_zones:
        if zone['from'] <= price <= zone['to']:
            return True
    return False
