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