import asyncio
import websockets
import json
from collections import deque
import threading
import time

class OrderFlowAnalyzer:
    def __init__(self, symbol, window_seconds=60):
        self.symbol = symbol.lower()
        self.trades = deque()
        self.window_seconds = window_seconds
        self.cvd = 0  # cumulative volume delta
        self.last_update = time.time()
        self.lock = threading.Lock()
        self._running = False
        self._ws = None

    def start(self):
        self._running = True
        thread = threading.Thread(target=self._run_async_loop, daemon=True)
        thread.start()

    def stop(self):
        self._running = False
        asyncio.run(self._close_ws())

    async def _close_ws(self):
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    def _run_async_loop(self):
        asyncio.run(self._main())

    async def _main(self):
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@trade"
        while self._running:
            try:
                async with websockets.connect(url) as ws:
                    self._ws = ws
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        now = time.time()
                        price = float(data['p'])
                        qty = float(data['q'])
                        is_buy = not data['m']
                        with self.lock:
                            self.trades.append({'ts': now, 'price': price, 'qty': qty, 'buy': is_buy})
                            if is_buy:
                                self.cvd += qty
                            else:
                                self.cvd -= qty
                            while self.trades and now - self.trades[0]['ts'] > self.window_seconds:
                                old = self.trades.popleft()
                                if old['buy']:
                                    self.cvd -= old['qty']
                                else:
                                    self.cvd += old['qty']
                            self.last_update = now
            except websockets.ConnectionClosedError as e:
                print(f'[{self.symbol}] Connection closed, reason: {e}. Переподключение через 5 секунд...')
                await asyncio.sleep(5)
            except Exception as e:
                print(f'[{self.symbol}] Ошибка: {e}. Переподключение через 5 секунд...')
                await asyncio.sleep(5)
            finally:
                await self._close_ws()

    def get_cvd(self):
        with self.lock:
            return self.cvd

    def get_recent_volume(self):
        with self.lock:
            return sum(t['qty'] for t in self.trades)

    def get_recent_buy_sell(self):
        with self.lock:
            buy = sum(t['qty'] for t in self.trades if t['buy'])
            sell = sum(t['qty'] for t in self.trades if not t['buy'])
            return buy, sell

    def get_info(self):
        buy, sell = self.get_recent_buy_sell()
        cvd = self.get_cvd()
        vol = self.get_recent_volume()
        return {
            'cvd': cvd,
            'buy_vol': buy,
            'sell_vol': sell,
            'vol': vol,
            'ts': self.last_update
        }