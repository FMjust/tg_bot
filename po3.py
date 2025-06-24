from datetime import time

def is_within_range(ts, start_hour, end_hour):
    return start_hour <= ts.hour < end_hour

def analyze_po3_structure(candles):
    if len(candles) < 24:
        return {"signal": False, "reason": "Недостаточно данных"}

    asia = []
    london = []
    ny = []

    for c in candles:
        ts = c["open_time"].time()
        if is_within_range(ts, 0, 8):
            asia.append(c)
        elif is_within_range(ts, 8, 13):
            london.append(c)
        elif is_within_range(ts, 13, 21):
            ny.append(c)

    if not asia or not london or not ny:
        return {"signal": False, "reason": "Нет всех сессий"}

    # 1. Азия: накопление = узкий диапазон
    asia_range = max(c["high"] for c in asia) - min(c["low"] for c in asia)
    if asia_range == 0:
        return {"signal": False, "reason": "Пустая Азия"}

    avg_asiarange = asia_range / len(asia)
    if avg_asiarange > 0.01 * asia[0]["close"]:
        return {"signal": False, "reason": "Азия слишком волатильна"}

    # 2. Лондон: манипуляция
    asia_high = max(c["high"] for c in asia)
    asia_low = min(c["low"] for c in asia)
    london_high = max(c["high"] for c in london)
    london_low = min(c["low"] for c in london)

    if london_high > asia_high * 1.001 and london_low < asia_low * 0.999:
        return {"signal": False, "reason": "Манипуляция в обе стороны"}

    direction = None
    if london_high > asia_high * 1.001:
        direction = "short"
    elif london_low < asia_low * 0.999:
        direction = "long"
    else:
        return {"signal": False, "reason": "Нет манипуляции в Лондоне"}

    # 3. Нью-Йорк: подтверждение BOS
    ny_close = ny[-1]["close"]
    ny_open = ny[0]["open"]
    ny_vol = ny[-1]["volume"]
    ny_volatility = ny[-1]["high"] - ny[-1]["low"]
    recent_volumes = [c["volume"] for c in candles[-10:-1]]
    avg_vol = sum(recent_volumes) / len(recent_volumes)

    if ny_vol < 0.5 * avg_vol:
        return {"signal": False, "reason": "Слабый объём NY"}

    if ny_volatility < 0.003 * ny_close:
        return {"signal": False, "reason": "Низкая волатильность NY"}

    if direction == "long" and ny_close > ny_open:
        entry = ny_close
        sl = london_low
        tp = entry + (entry - sl) * 2
    elif direction == "short" and ny_close < ny_open:
        entry = ny_close
        sl = london_high
        tp = entry - (sl - entry) * 2
    else:
        return {"signal": False, "reason": "Нет подтверждения BOS в NY"}

    return {
        "signal": True,
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "reason": "PO3 Session model: Asia-London-NY",
        "ob": False,                           
    }
