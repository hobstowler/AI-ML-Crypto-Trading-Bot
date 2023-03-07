import os
import sys
import time
import datetime
from pprint import pprint

import numpy as np
import requests
import pandas as pd
from google.cloud import storage
from datetime import datetime, timezone, timedelta

from RL_Model.classes.agent import Agent
from RL_Model.classes.environment import StockTradingEnv
from RL_Model.data.binance_api import BinanceAPI
from RL_Model.data.datastore_wrapper import DatastoreWrapper
from RL_Model.misc.secret_client import SecretClient


class RL:
    def __init__(self):
        self.training_csv = 'train_2021-01-01-2021-12-31_1d.csv'
        self.price_hist_csv = 'rl_price_hist_1h'
        self.actor_model_name = 'actor_ppo_1h'
        self.critic_model_name = 'critic_ppo_1h'
        self.bucket_name = 'ai-ml-bitcoin-bot.appspot.com'
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        self.df = None
        self.balances = {}

        # Hyper Params
        self.batch_size = 10
        self.n_epochs = 5
        self.alpha = 0.0003

        self.kill_thresh = 0.6
        self.initial_account_balance = 10000
        self.trading_costs_rate = 0.001
        self.max_open_positions = 1
        self.percent_capital = 0.1

        self.secret_client = SecretClient()
        self.binance_secret = self.secret_client.get_secret("binance-api-secret-hobs")
        self.binance_key = self.secret_client.get_secret("binance-api-key-hobs")

        self.binance_client = BinanceAPI(self.binance_key, self.binance_secret)
        self.datastore_client = DatastoreWrapper()

    def initialize_actor(self):
        pass

    def check_for_models(self):
        actor_file_path = os.path.join(os.getcwd(), 'models', self.actor_model_name)
        critic_file_path = os.path.join(os.getcwd(), 'models', self.actor_model_name)
        if os.path.exists(actor_file_path) and os.path.exists(critic_file_path):
            return

        if not os.path.exists(actor_file_path):
            blob = self.bucket.blob(self.actor_model_name)
            if blob.exists():
                blob.download_to_filename(actor_file_path)
            else:
                # TODO
                raise FileNotFoundError

        if not os.path.exists(critic_file_path):
            blob = self.bucket.blob(self.critic_model_name)
            if blob.exists():
                blob.download_to_filename(critic_file_path)
            else:
                # TODO
                raise FileNotFoundError

    def generate_price_hist_csv(self):
        columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time',
                   'asset volume', 'num trades', 'base asset volume', 'quote asset volume', 'meh']
        interval_time = 60 * 60 * 1000
        interval = '1h'
        symbol = 'BTCUSD'
        start_dt = int((datetime.utcnow() - timedelta(31)).timestamp() * 1000)  # 31 days for padding...
        end_dt = int(datetime.utcnow().timestamp() * 1000)
        max_candles = 1000

        dataframes = []
        failures = 0
        for i in range(start_dt, end_dt, interval_time * max_candles):
            start = i
            end = i + (interval_time * max_candles) - interval_time
            end = end if end <= end_dt else end_dt
            resp = requests.get(
                f'https://api.binance.us/api/v3/klines?symbol='
                f'{symbol}&interval={interval}&limit={max_candles}&startTime={start}&endTime={end}')
            if resp.status_code == 200:
                df = pd.DataFrame(resp.json(), columns=columns)
                dataframes.append(df)
                failures = 0
                # df = df.append(ndf, ignore_index=False)
            elif resp.status_code == 429:
                print(f'error 429: rate limit tripped at i={i}. start: {start}, end: {end}')
                print(resp.json())
                time.sleep(10)
                failures += 1
                i -= 1  # retry
            else:
                failures += 1

            if failures == 5:
                print('Failed 5 times in a row. Raising Exception.')
                raise Exception

        df = pd.concat(dataframes, ignore_index=True)

        # Push to cloud for later
        blob = self.bucket.blob(self.price_hist_csv)
        with blob.open("w") as f:
            f.write(df.to_csv(index=False))

        # Write locally to csv for now
        df.to_csv(os.path.join(os.getcwd(), 'data', f'{self.price_hist_csv}.csv'), index=False)

    def check_for_price_hist_csv(self):
        """
        Checks for a local copy of the price history CSV and either downloads it or generates it from binance data
        :return:
        """
        file_path = os.path.join(os.getcwd(), 'data', f'{self.price_hist_csv}.csv')
        if not os.path.exists(file_path):
            blob = self.bucket.blob(self.price_hist_csv)
            if blob.exists():
                with blob.open("r") as f:
                    df = pd.read_csv(f)
                df.to_csv(file_path, index=False)
            else:
                self.generate_price_hist_csv()

        return #df

    def check_files(self):
        self.check_for_price_hist_csv()
        self.check_for_models()

    def initialize_data(self):
        self.check_files()
        self.get_binance_state()
        self.get_quote()

    def get_binance_state(self):
        balances = self.binance_client.get_account_balances(['BTC', 'USDT'])
        self.balances = {
            'BTC': list(filter(lambda l: l['asset'] == 'BTC', balances))[0]['free'],
            'USDT': list(filter(lambda l: l['asset'] == 'USDT', balances))[0]['free']
        }

    def get_quote(self):
        client = self.binance_client.get_client()
        kline = client.get_klines(**{'symbol': 'BTCUSDT', 'limit': 1, 'interval': client.KLINE_INTERVAL_1HOUR})
        self.df.loc[len(self.df)] = kline[0]

        # update local csv
        self.df.to_csv(os.path.join(os.getcwd(), 'data', f'{self.price_hist_csv}.csv'), index=False)

        # update Cloud Storage
        blob = self.bucket.blob(self.price_hist_csv)
        with blob.open("w") as f:
            f.write(self.df.to_csv(index=False))


    def initialize_models(self):
        self.env = StockTradingEnv(self.df, self.initial_account_balance, self.trading_costs_rate,
                              self.max_open_positions, self.percent_capital, self.kill_thresh)
        self.agent = Agent(n_actions=self.env.action_space.n, batch_size=self.batch_size,
                      alpha=self.alpha, n_epochs=self.n_epochs,
                      input_dims=self.env.observation_space.shape,
                      actor_file=f'actor_ppo_1h',
                      critic_file=f'critic_ppo_1h')

    def make_observation(self):
        step = len(self.df)
        close = self.df.loc[step, 'Close'].item()
        volume = self.df.loc[step, 'Volume'].item()
        sma30 = self.df.loc[step, 'SMA30'].item()
        ema30 = self.df.loc[step, 'EMA30'].item()
        cma = self.df.loc[step, 'CMA'].item()
        bollinger_upper = self.df.loc[step, 'bollinger_upper'].item()
        bollinger_lower = self.df.loc[step, 'bollinger_lower'].item()
        macd = self.df.loc[step, 'macd'].item()
        signal = self.df.loc[step, 'signal'].item()
        rs = self.df.loc[step, 'rs'].item()

        obs = np.array([close, volume, sma30, ema30, cma, bollinger_lower,
                        bollinger_upper, macd, signal, rs, 0.5])
        return obs

    def make_decision(self, observation):
        action, _, _ = self.agent.choose_action(observation)

        if action == 0 and \
                self.balances['USDT'] > self.initial_account_balance * self.kill_thresh and \
                self.balances['BTC'] == 0.0:
            amount = self.balances['USDT'] * self.percent_capital
            quantity = amount / self.df.loc['_Close'][-1:]
            print(quantity)
            #response = self.binance_client.buy_asset('BTCUSDT', quantity)
        elif action == 1 and self.balances['BTC'] > 0:
            print(self.balances['BTC'])
            #response = self.binance_client.sell_asset('BTCUSDT', self.balances['BTC'])

        self.get_binance_state()

    def run(self):
        self.initialize_data()
        self.initialize_models()
        obs = self.make_observation()
        self.make_decision(obs)

    def train(self):
        pass


if __name__ == "__main__":
    print(sys.argv)
    args = sys.argv[1:]
    print(args)
    """rl_runner = RL()
    rl_runner.run()
    secret_client = SecretClient()
    binance_secret = secret_client.get_secret("binance-api-secret-hobs")
    binance_key = secret_client.get_secret("binance-api-key-hobs")
    ba = BinanceAPI(binance_key, binance_secret)
    #print(ba.get_account_information())
    pprint(ba.get_account_balances())
    balances = ba.get_account_balances(['BTC', 'USDT'])
    print(list(filter(lambda l: l['asset'] == 'BTC', balances))[0]['free'])
    print(ba.get_client().get_symbol_ticker(**{'symbol': 'BTCUSDT'}))
    print(ba.get_client().get_klines(**{'symbol': 'BTCUSDT', 'limit': 1, 'interval': ba.get_client().KLINE_INTERVAL_1HOUR}))

"""