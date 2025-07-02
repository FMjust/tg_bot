def get_trend_htf(candles):
    """
    Определение тренда старшего таймфрейма.
    candles: [{'open': ..., 'high': ..., 'low': ..., 'close': ...}, ...]
    Алгоритм: если последний Close > предыдущего, тренд лонг, иначе шорт.
    """
    closes = [c['close'] for c in candles[-3:]]
    return 'long' if closes[-1] > closes[-2] else 'short'

def get_pd_zones(candles):
    """
    Определяет Premium/Discount зоны по последним 20 свечам.
    Premium: верхняя половина диапазона, Discount: нижняя.
    Возвращает кортежи (premium_zone, discount_zone): (min, mid), (mid, max)
    """
    highs = [c['high'] for c in candles[-20:]]
    lows = [c['low'] for c in candles[-20:]]
    swing_high = max(highs)
    swing_low = min(lows)
    pd_middle = (swing_high + swing_low) / 2
    premium = (pd_middle, swing_high)
    discount = (swing_low, pd_middle)
    return premium, discount

def filter_signals_by_htf(signals, htf_trend):
    """
    Оставляет только сигналы, совпадающие с трендом старшего ТФ.
    signals: [{'direction': 'long'/'short', ...}]
    """
    return [s for s in signals if s and s.get('direction') == htf_trend]
def find_fvg_zones(candles):
    """
    Находит FVG (Fair Value Gap) по классике:
    - Если между high предыдущей и low следующей свечи есть 'дырка', фиксируем зону.
    """
    fvg_zones = []
    for i in range(1, len(candles) - 1):
        prev = candles[i - 1]
        curr = candles[i]
        nxt = candles[i + 1]
        if curr['low'] > prev['high']:  # FVG вверх
            fvg_zones.append({'type': 'fvg', 'direction': 'long', 'from': prev['high'], 'to': curr['low']})
        elif curr['high'] < prev['low']:  # FVG вниз
            fvg_zones.append({'type': 'fvg', 'direction': 'short', 'from': curr['high'], 'to': prev['low']})
    return fvg_zones

def find_order_blocks(candles, direction):
    """
    Находит Order Blocks (последняя противоположная свеча перед импульсом BOS).
    Упрощённо: ищем большие свечи с сильным объёмом перед движением в нужную сторону.
    """
    obs = []
    for i in range(1, len(candles) - 1):
        c = candles[i]
        if direction == 'long' and c['close'] < c['open']:  # медвежья свеча перед ростом
            obs.append({'type': 'ob', 'direction': 'long', 'from': c['low'], 'to': c['high']})
        elif direction == 'short' and c['close'] > c['open']:  # бычья свеча перед падением
            obs.append({'type': 'ob', 'direction': 'short', 'from': c['low'], 'to': c['high']})
    return obs

def find_poi_zones(candles, pd_zone, direction):
    """
    Ищем POI (точки входа) в Discount (лонг) или Premium (шорт).
    Приоритет: FVG -> OB -> ликвидность (equal highs/lows).
    """
    pois = []
    fvg = [z for z in find_fvg_zones(candles) if pd_zone[0] <= z['from'] <= pd_zone[1]]
    ob = [z for z in find_order_blocks(candles, direction) if pd_zone[0] <= z['from'] <= pd_zone[1]]

    pois.extend(fvg)
    pois.extend(ob)
    # Ликвидность (равные экстремумы) — можно добавить отдельной функцией при необходимости

    # Сортируем: сначала FVG, потом OB
    pois = sorted(pois, key=lambda z: z['from'])
    return pois

def find_fta_zones(candles, direction, pd_zone):
    """
    Первая зона на пути к цели (FTA) — ближайший противоположный FVG или OB.
    """
    opp_direction = 'short' if direction == 'long' else 'long'
    # FVG и OB в зоне тейка (Premium — если long, Discount — если short)
    fvg = [z for z in find_fvg_zones(candles) if pd_zone[0] <= z['from'] <= pd_zone[1] and z['direction'] == opp_direction]
    ob = [z for z in find_order_blocks(candles, opp_direction) if pd_zone[0] <= z['from'] <= pd_zone[1]]

    ftas = []
    ftas.extend(fvg)
    ftas.extend(ob)
    ftas = sorted(ftas, key=lambda z: z['from'])
    return ftas