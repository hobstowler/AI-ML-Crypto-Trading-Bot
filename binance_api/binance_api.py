from binance.client import Client
from .binance_keys import BinanceKeys


class BinanceAPI:
    
    def __init__(self) -> None:
        self._keys = BinanceKeys()
        self._client = Client(self._keys.get_api_key(), self._keys.get_api_secret(), testnet=True)

    def print_exchange_info(self):
        print(self._client.get_exchange_info())
        
    def print_symbol_info(self, symbol: str):
        print(self._client.get_symbol_info(symbol))

    def print_ticker_info(self, ticker_symbol: str):
        tickers = self._client.get_ticker()
        for ticker in tickers:
            if ticker["symbol"] == ticker_symbol:
                print(ticker)
