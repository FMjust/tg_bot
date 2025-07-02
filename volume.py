def average_volume(candles, n=20):
    """
    Средний объём за n последних свечей.
    """
    vols = [c['volume'] for c in candles[-n:]]
    return sum(vols) / len(vols) if vols else 0

def is_volume_spike(candle, candles, coef=1.5, n=20):
    """
    True если объём этой свечи выше среднего в n раз.
    """
    avg_vol = average_volume(candles, n)
    return candle['volume'] > avg_vol * coef

def find_volume_clusters(candles, coef=2.0, n=40):
    """
    Находит свечи-кластеры — экстремальные всплески объёма.
    coef — во сколько раз объём должен быть выше среднего (например, 2.0)
    n — "окно" для средней
    Возвращает список словарей: [{'i': индекс, 'volume': ..., 'high': ..., 'low': ...}, ...]
    """
    clusters = []
    avg_vol = average_volume(candles, n)
    window = candles[-n:] if len(candles) >= n else candles
    for i, c in enumerate(window):
        if c['volume'] > avg_vol * coef:
            clusters.append({'i': i, 'volume': c['volume'], 'high': c['high'], 'low': c['low']})
    return clusters

def filter_zones_by_volume(zones, candles, coef=1.5, n=20, strict_overlap=False):
    """
    Фильтрует зоны (FVG/OB): оставляет только те, где был всплеск объёма.
    strict_overlap=False — ищет любую свечу, где зона пересекается с high/low свечи.
    strict_overlap=True — зона должна полностью быть внутри тела свечи (жёстче).
    """
    filtered = []
    for z in zones:
        # Пробегаем свечи, ищем где зона появилась (можно сделать жёсткую или мягкую фильтрацию)
        for c in candles:
            # "Мягкая" фильтрация: хотя бы частичное пересечение зоны и диапазона свечи
            if not strict_overlap and (z['from'] <= c['high'] and z['to'] >= c['low']):
                if is_volume_spike(c, candles, coef=coef, n=n):
                    filtered.append(z)
                    break
            # "Жёсткая": зона полностью внутри high/low свечи
            elif strict_overlap and (z['from'] >= c['low'] and z['to'] <= c['high']):
                if is_volume_spike(c, candles, coef=coef, n=n):
                    filtered.append(z)
                    break
    return filtered

def zone_has_cluster(z, clusters):
    """
    Проверяет, есть ли кластер (volume spike) внутри зоны z.
    """
    for cl in clusters:
        if z['from'] <= cl['high'] and z['to'] >= cl['low']:
            return True
    return False