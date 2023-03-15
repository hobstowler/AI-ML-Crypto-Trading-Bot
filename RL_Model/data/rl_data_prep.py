import time

import pandas as pd
import numpy as np


class RLDataPrepper:
    def __init__(self, interval, file: str = None, df: pd.DataFrame = None):
        if file is not None:
            df = pd.read_csv(file)
        elif df is None:
            raise Exception
        self.interval = interval
        self.original = df
        self.df = df[['Close', 'asset volume']]
        #self.df = self.df.drop(columns=['Volume'], axis=1)
        self.df = self.df.rename(columns={'asset volume': 'Volume'})
        self.df = self.df.loc[(self.df != 0.0).any(axis=1)]
        if self.interval == '1m':
            self.int_per_day = 1440
        elif self.interval == '5m':
            self.int_per_day = 288
        elif self.interval == '30m':
            self.int_per_day = 48
        elif self.interval == '1h':
            self.int_per_day = 24
        else:
            self.int_per_day = 1

    def reset(self):
        self.df = self.original

    def do_it_all(self, normalize: bool = True, last: bool = False) -> pd.DataFrame:
        self.diffs(last=last)
        self.generate_sma30(30 * self.int_per_day, last=last)
        self.generate_ema30(30 * self.int_per_day, last=last)
        self.generate_cma(last=last)
        self.bollinger_bands(last=last)
        self.macd(last=last)
        self.rsi(last=last)
        if normalize:
            self.normalize()

            self.df['_Volume'] = self.original['asset volume']
            self.df['_Close'] = self.original['Close']
        self.df = self.df.replace(np.inf, np.nan)
        self.df = self.df.dropna()
        self.df = self.df.reset_index(drop=True)

        return self.df

    def diffs(self, last: bool = False):
        if last:
            self.df['Close Rt'][-1:] = self.df['Close'][-2:].pct_change()
        else:
            self.df['Close Rt'] = self.df['Close'].pct_change()

    def generate_sma30(self, window: int = 30, last: bool = False):
        self.df[f'SMA30'] = self.df['Close'].rolling(window).mean()

    def generate_cma(self, last: bool = False):
        self.df[f'CMA'] = self.df['Close'].expanding().mean()

    def generate_ema30(self, window: int = 30, last: bool = False):
        self.df[f'EMA30'] = self.df['Close'].ewm(span=window).mean()

    def bollinger_bands(self, window: int = 30, last: bool = False):
        window = window * self.int_per_day
        self.generate_sma30(window)
        rstd = self.df['Close'].rolling(window).std()
        self.df['bollinger_upper'] = self.df[f'SMA30'] + 2 * rstd
        self.df['bollinger_lower'] = self.df[f'SMA30'] - 2 * rstd

    def macd(self, last: bool = False):
        exp1 = self.df['Close'].ewm(span=(12 * self.int_per_day)).mean()
        exp2 = self.df['Close'].ewm(span=(26 * self.int_per_day)).mean()
        self.df['macd'] = exp1 - exp2
        self.df['signal'] = self.df['macd'].ewm(span=(9 * self.int_per_day)).mean()

    def rsi(self, last: bool = False):
        delta = self.df['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=(13 * self.int_per_day)).mean()
        ema_down = down.ewm(com=(13 * self.int_per_day)).mean()
        self.df['rs'] = ema_up / ema_down

    def normalize(self, last: bool = False):
        for column in self.df.columns:
            self.df[column] = self.df[column] / self.df[column].abs().max()


if __name__ == '__main__':
    dp = RLDataPrepper('1h', file='../../data/train_2021-01-01-2021-12-31_1h.csv')
    start = time.perf_counter()
    df = dp.do_it_all()
    end = time.perf_counter()
    print(f'done in {end - start:0.2f} seconds')
    print(df.head())