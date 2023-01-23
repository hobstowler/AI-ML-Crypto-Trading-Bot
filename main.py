from binance_api.binance_api import BinanceAPI

if __name__ == "__main__":
    binance = BinanceAPI()
    binance.print_ticker_info("BTCUSDT")