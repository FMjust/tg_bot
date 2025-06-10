
def score_signal(ob, fvg, multi_bos, rsi_ok, near_liq, fvg_gap=False):
    score = 0
    components = []

    if ob:
        score += 1
        components.append("☑️ OB")
    else:
        components.append("❌ OB")

    if fvg:
        score += 1
        components.append("☑️ FVG")
    else:
        components.append("❌ FVG")

    if multi_bos:
        score += 1
        components.append("☑️ Multi-BOS")
    else:
        components.append("❌ Multi-BOS")

    if rsi_ok:
        score += 1
        components.append("☑️ RSI OK")
    else:
        components.append("❌ RSI")

    if near_liq:
        score += 1
        components.append("☑️ Liquidity OK")
    else:
        components.append("❌ Liquidity")

    if fvg_gap:
        score += 1
        components.append("☑️ Entry вне GAP")
    else:
        components.append("❌ GAP")

    summary = f"📊 Сила сигнала: {score} из 6\n" + "\n".join(components)
    return score, summary
