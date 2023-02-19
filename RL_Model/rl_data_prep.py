import pandas as pd
import numpy as np


class RLDataPrepper:
    def __init__(self, file):
        df = pd.read_csv(file)
        self.original = df
        self.df = df

    def reset(self):
        self.df = self.original

    def do_it_all(self) -> pd.DataFrame:
        self.generate_sma()
        self.generate_ema()
        self.generate_cma()
        self.bollinger_bands()
        self.macd()
        self.rsi()
        self.vwap()
        self.df.dropna(inplace=True)

        return self.df

    def generate_sma(self, window: int = 30):
        self.df[f'SMA{window}'] = self.df['Close'].rolling(window).mean()

    def generate_cma(self):
        self.df[f'CMA'] = self.df['Close'].expanding().mean()

    def generate_ema(self, window: int = 30):
        self.df[f'EMA{window}'] = self.df['Close'].ewm(span=window).mean()

    def bollinger_bands(self, window: int = 30):
        self.generate_sma(window)
        rstd = self.df['Close'].rolling(window).std()
        self.df['bollinger_upper'] = self.df[f'SMA{window}'] + 2 * rstd
        self.df['bollinger_lower'] = self.df[f'SMA{window}'] - 2 * rstd

    def macd(self):
        exp1 = self.df['Close'].ewm(span=12).mean()
        exp2 = self.df['Close'].ewm(span=26).mean()
        self.df['macd'] = exp1 - exp2
        self.df['signal'] = self.df['macd'].ewm(span=9).mean()

    def rsi(self):
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13).mean()
        ema_down = down.ewm(com=13).mean()
        self.df['rs'] = ema_up / ema_down

    def vwap(self):
        self.df['vwap']