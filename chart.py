
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

def plot_signal_chart(candles, signal, symbol="BTCUSDT", filename="chart.png"):
    times = [c["open_time"] for c in candles]
    opens = [c["open"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"{symbol} — Smart Money Setup", fontsize=12)

    for i in range(len(candles)):
        color = "green" if closes[i] >= opens[i] else "red"
        ax.plot([times[i], times[i]], [lows[i], highs[i]], color=color, linewidth=1)
        ax.plot([times[i], times[i]], [opens[i], closes[i]], color=color, linewidth=4)

    # Вход
    ax.axhline(signal["entry"], color="blue", linestyle="--", label="Entry")
    # SL
    ax.axhline(signal["sl"], color="red", linestyle="--", label="SL")
    # TP
    ax.axhline(signal["tp"], color="green", linestyle="--", label="TP")

    # Подписи
    ax.text(times[-1], signal["entry"], " Entry", color="blue", va="bottom", fontsize=9)
    ax.text(times[-1], signal["sl"], " SL", color="red", va="bottom", fontsize=9)
    ax.text(times[-1], signal["tp"], " TP", color="green", va="bottom", fontsize=9)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
