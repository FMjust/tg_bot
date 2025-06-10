
import csv
from datetime import datetime

def log_signal_to_csv(
    symbol, direction, entry, sl, tp,
    ob, fvg, multi_bos, rsi, near_liq, strategy, filename="signals_log.csv"
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = [
        now, symbol, direction, round(entry, 2), round(sl, 2), round(tp, 2),
        int(ob), int(fvg), int(multi_bos), round(rsi, 2), int(near_liq), strategy
    ]
    header = [
        "datetime", "symbol", "direction", "entry", "sl", "tp",
        "ob", "fvg", "multi_bos", "rsi", "near_liq", "strategy"
    ]
    try:
        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f"Ошибка при логировании: {e}")
