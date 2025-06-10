def detect_liquidity_sweep_and_bos(candles, sensitivity=3):
    if len(candles) < sensitivity + 10:
        return {"signal": False, "reason": "Недостаточно данных", "strategy": "Liquidity + BOS"}

    last = candles[-1]
    prev = candles[-2]
    recent = candles[-sensitivity-1:-1]
    volume_check = [c["volume"] for c in candles[-10:-1]]
    avg_volume = sum(volume_check) / len(volume_check)

    volatility = last["high"] - last["low"]
    if volatility < 0.003 * last["close"]:
        return {"signal": False, "reason": "Слабая волатильность", "strategy": "Liquidity + BOS"}

    if last["volume"] < 0.5 * avg_volume:
        return {"signal": False, "reason": "Слабый объём", "strategy": "Liquidity + BOS"}

    highs = [c["high"] for c in recent]
    lows = [c["low"] for c in recent]
    avg_high = sum(highs) / len(highs)
    avg_low = sum(lows) / len(lows)

    sweep_high = last["high"] > avg_high * 1.0005 and last["close"] < avg_high
    sweep_low = last["low"] < avg_low * 0.9995 and last["close"] > avg_low

    bos_up = sweep_low and last["close"] > prev["high"]
    bos_down = sweep_high and last["close"] < prev["low"]

    ob_confirmed = False
    ob_zone = None

    if bos_up:
        for candle in reversed(candles[:-1]):
            if candle["close"] < candle["open"]:
                ob_zone = (candle["open"], candle["close"])
                if ob_zone[0] <= last["low"] <= ob_zone[1]:
                    ob_confirmed = True
                break

        return {
            "signal": True,
            "direction": "long",
            "entry": last["close"],
            "sl": last["low"],
            "tp": last["close"] + (last["close"] - last["low"]) * 2,
            "ob": ob_confirmed,
            "reason": "Liquidity sweep + BOS вверх",
            "strategy": "Liquidity + BOS"
        }

    elif bos_down:
        for candle in reversed(candles[:-1]):
            if candle["close"] > candle["open"]:
                ob_zone = (candle["close"], candle["open"])
                if ob_zone[1] <= last["high"] <= ob_zone[0]:
                    ob_confirmed = True
                break

        return {
            "signal": True,
            "direction": "short",
            "entry": last["close"],
            "sl": last["high"],
            "tp": last["close"] - (last["high"] - last["close"]) * 2,
            "ob": ob_confirmed,
            "reason": "Liquidity sweep + BOS вниз",
            "strategy": "Liquidity + BOS"
        }

    return {"signal": False, "reason": "Сигнал не найден", "strategy": "Liquidity + BOS"}

def detect_po3_amd_model(candles):
    if len(candles) < 15:
        return {"signal": False, "reason": "Недостаточно данных для PO3", "strategy": "PO3 + AMD Model"}

    range_high = max(c["high"] for c in candles[-15:-5])
    range_low = min(c["low"] for c in candles[-15:-5])
    range_mid = (range_high + range_low) / 2

    manipulation_candle = candles[-4]
    broke_above = manipulation_candle["high"] > range_high * 1.001
    broke_below = manipulation_candle["low"] < range_low * 0.999

    last = candles[-1]
    bos = False
    direction = None

    if broke_above and last["close"] < range_mid:
        bos = True
        direction = "short"
    elif broke_below and last["close"] > range_mid:
        bos = True
        direction = "long"

    if bos:
        entry = last["close"]
        sl = entry * (0.99 if direction == "short" else 1.01)
        tp = entry * (0.97 if direction == "short" else 1.03)

        return {
            "signal": True,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "direction": direction,
            "reason": "PO3: Manipulation + BOS + AMD",
            "ob": False,
            "strategy": "PO3 + AMD Model"
        }

    return {"signal": False, "reason": "PO3 не подтверждён", "strategy": "PO3 + AMD Model"}
