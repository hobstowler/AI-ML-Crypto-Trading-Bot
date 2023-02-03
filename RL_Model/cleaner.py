import pandas as pd
import numpy as np

class Cleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean(self, dropnna=True):
        if dropnna:
            self.df.dropna()

    def calc_pct_change(self, **cols):
        """
        Takes multiple key/value arguments to calculate the percent change. The target column name will be the value of
        the key and the resulting column the value of the value. There is no return, but the dataframe associated with
        the cleaner class will be updated.
        """
        for key, value in cols.items():
            self.df[value] = self.df[key].pct_change()

    def calc_log_returns(self, **cols):
        for key, value in cols.items():
            self.df[value] = np.log(self.df[key] / self.df[key].shift(1))

    def cumsum_log_returns(self, **cols):
        for key, value in cols.items():
            self.df[value] = self.df[key].cumsum()

    def normalize_log_returns(self, **cols):
        for key, value in cols.items():
            self.df[value] = np.exp(self.df[key]) - 1

    def add_target(self, col):
        self.df.loc[self.df[col].shift(-1) > self.df[col], 'TARGET'] = 1
        self.df.loc[self.df[col].shift(-1) <= self.df[col], 'TARGET'] = -1

    def one_hot(self, col):
        one_hot = pd.get_dummies(self.df[col], drop_first=True)
        self.df = self.df.join(one_hot)