import requests
from datetime import datetime

def get_candles(symbol="BTCUSDT", interval="15m", limit=50):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    try:
        data = response.json()
    except Exception:
        return []

    if not isinstance(data, list):
        print(f"[!] Ошибка по символу {symbol}: {data}")
        return []

    candles = []
    for k in data:
        candles.append({
            "open_time": datetime.fromtimestamp(int(k[0]) / 1000),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        })
    return candles