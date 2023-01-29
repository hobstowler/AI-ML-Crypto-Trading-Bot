from binance.client import Client
from .binance_keys import BinanceKeys
import datetime
import pandas 

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
                
    def get_kline_intervals(self):
        """Returns a dict of possible KLINE intervals from Binance in length of minutes.

        Returns:
            (dict): hash of possible KLINE intervals whose keys are in minutes
        """
        return {
            1: self._client.KLINE_INTERVAL_1MINUTE,
            3: self._client.KLINE_INTERVAL_3MINUTE,
            5: self._client.KLINE_INTERVAL_5MINUTE,
            15: self._client.KLINE_INTERVAL_15MINUTE,
            30: self._client.KLINE_INTERVAL_30MINUTE,
            60: self._client.KLINE_INTERVAL_1HOUR,
            120: self._client.KLINE_INTERVAL_2HOUR,
            240: self._client.KLINE_INTERVAL_4HOUR,
            360: self._client.KLINE_INTERVAL_6HOUR,
            480: self._client.KLINE_INTERVAL_8HOUR,
            720: self._client.KLINE_INTERVAL_12HOUR,
            1440: self._client.KLINE_INTERVAL_1DAY,
            4320: self._client.KLINE_INTERVAL_3DAY,
            10080: self._client.KLINE_INTERVAL_1WEEK,
            "month": self._client.KLINE_INTERVAL_1MONTH
        }

    def get_historical_data_candles(
        self, ticker_symobol: str, minutes_from_now: int, time_interval_in_minutes: int):
        """Retrieves past candlestick data for `ticker_symbol`.

        Args:
            ticker_symobol (str): e.g. `"BTCUSDT"`.
            minutes_from_now (int): quantity of minutes in the past to pull data from.
            time_interval_in_minutes (int): candlestick interval (e.g. `1` for 1 minute candles)

        Raises:
            ValueError: for incorrect argument values.

        Returns:
            (list): list of OHLCV values from Binance.
        """
        if minutes_from_now < 1:
            raise ValueError("minutes_from_now must be larger that 0.")
        time_interval = self.get_kline_intervals().get(time_interval_in_minutes)
        if time_interval is None:
            raise ValueError("time_interval_in_minutes is invalid.")
        current_time = datetime.datetime.now()
        past_time = current_time - datetime.timedelta(minutes=minutes_from_now)
        return self._client.get_historical_klines(
            ticker_symobol, time_interval, str(past_time), str(current_time))


    def get_candlestick_dataframe(
        self, ticker_symbol: str, minutes_from_now: int, time_inteval_in_minutes: int):
        """Gets a Pandas-labelled dataframe of candlestick data

        Args:
            ticker_symobol (str): e.g. `"BTCUSDT"`.
            minutes_from_now (int): quantity of minutes in the past to pull data from.
            time_interval_in_minutes (int): candlestick interval (e.g. `1` for 1 minute candles)

        Returns:
            (Pandas.DataFrame): Two-dimensional, size-mutable, tabular data of requested candles.
        """
        candles = self.get_historical_data_candles(
            ticker_symbol, minutes_from_now, time_inteval_in_minutes)
        candles_dataframe = pandas.DataFrame(
            candles, 
            columns=[
                "date_time", "open_price", "high_price", "low_price", "close_price", "volume", 
                "close_time", "quote_asset_volume", "qty_transactions", 
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        candles_dataframe.date_time = pandas.to_datetime(candles_dataframe.date_time, unit='ms')
        candles_dataframe.date_time.dt.strftime("%Y/%m/%d %H:%M:%S")
        candles_dataframe.set_index('date_time', inplace=True)
        candles_dataframe.drop(["ignore"], axis=1)
        return candles_dataframe
    
    def export_candlestick_dataframe_csv(
        self, pandas_dataframe: pandas.DataFrame, csv_filepath: str):
        """Exports a Pandas DataFrame to csv file format.

        Args:
            pandas_dataframe (pandas.DataFrame): data frame to export
            csv_filepath (str): filepath to save csv.
        """
        pandas_dataframe.to_csv(csv_filepath)
        return