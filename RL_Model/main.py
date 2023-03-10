import math
import os
import pickle
import time
import datetime

import numpy as np
import requests
import pandas as pd
from flask import Flask, jsonify, request
from google.cloud import storage
from datetime import datetime, timezone, timedelta

from classes.agent import Agent
from classes.environment import StockTradingEnv
from data.binance_api import BinanceAPI
from data.datastore_wrapper import DatastoreWrapper
from data.rl_data_prep import RLDataPrepper
from misc.secret_client import SecretClient


app = Flask(__name__, instance_relative_config=True)


class RL:
    def __init__(self):
        self.training_csv = 'train_2021-01-01-2021-12-31_1d.csv'
        self.price_hist_csv = 'rl_price_hist_1h'
        self.actor_model_name = 'actor_ppo_1d'
        self.critic_model_name = 'critic_ppo_1d'
        self.bucket_name = 'ai-ml-bitcoin-bot.appspot.com'
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        self.df = None
        self.balances = {}

        # Hyper Params
        self.batch_size = 10
        self.n_episodes = 500
        self.n_epochs = 5
        self.alpha = 0.0003
        self.N = 20

        self.kill_thresh = 0.6
        self.initial_account_balance = 10000.0
        self.trading_costs_rate = 0.001
        self.max_open_positions = 1
        self.percent_capital = 0.1

        self.secret_client = SecretClient()
        self.binance_secret = self.secret_client.get_secret("binance-api-secret-hobs")
        self.binance_key = self.secret_client.get_secret("binance-api-key-hobs")

        self.binance_client = BinanceAPI(self.binance_key, self.binance_secret)
        self.datastore_client = DatastoreWrapper()

        self.state = self.load_state()

    def load_state(self):
        state_file_path = os.path.join(os.getcwd(), 'state.p')
        if os.path.exists(state_file_path):
            with open(state_file_path, "rb") as f:
                state = pickle.load(f)
            return state
        else:
            state = {
                "session_id": self.datastore_client.create_session(**{"session_name": "RL Model Spot Trading",
                                                                      "type": "live (spot)",
                                                                      "model_name": "RL Model",
                                                                      "starting_balance": self.initial_account_balance,
                                                                      "starting_coins": 1.0,
                                                                      "crypto_type": "BTC"}),
                "step": 0
            }
            self.save_state()
            return state

    def save_state(self):
        state_file_path = os.path.join(os.getcwd(), 'state.p')
        with open(state_file_path, "wb") as f:
            pickle.dump(self.state, f)


    def run(self):
        self.initialize_data()
        print("init models")
        self.initialize_models()
        print("making observation")
        obs = self.make_observation()
        print("making decision")
        self.make_decision(obs)

    def initialize_data(self):
        self.check_files()
        self.get_binance_state()
        self.get_quote()

        dp = RLDataPrepper('1h', df=self.df)
        self.df = dp.do_it_all()

    def initialize_models(self):
        self.env = StockTradingEnv(self.df, self.initial_account_balance, self.trading_costs_rate,
                              self.max_open_positions, self.percent_capital, self.kill_thresh)
        self.agent = Agent(n_actions=self.env.action_space.n, batch_size=self.batch_size,
                      alpha=self.alpha, n_epochs=self.n_epochs,
                      input_dims=self.env.observation_space.shape,
                      actor_file=self.actor_model_name,
                      critic_file=self.critic_model_name)

    def check_files(self):
        self.check_for_price_hist_csv()
        self.check_for_models()

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

        self.df = df

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
                self.df = df
            else:
                self.generate_price_hist_csv()
        else:
            self.df = pd.read_csv(file_path)

        return #df

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

    def get_binance_state(self):
        balances = self.binance_client.get_account_balances(['BTC', 'USDT'])
        self.balances = {
            'BTC': float(list(filter(lambda l: l['asset'] == 'BTC', balances))[0]['free']),
            'USDT': float(list(filter(lambda l: l['asset'] == 'USDT', balances))[0]['free'])
        }

    def get_quote(self):
        client = self.binance_client.get_client()
        kline = client.get_klines(**{'symbol': 'BTCUSDT', 'limit': 1, 'interval': client.KLINE_INTERVAL_1HOUR})
        mod_kline = [
            int(kline[0][0]),
            float(kline[0][1]),
            float(kline[0][2]),
            float(kline[0][3]),
            float(kline[0][4]),
            float(kline[0][5]),
            int(kline[0][6]),
            float(kline[0][7]),
            int(kline[0][8]),
            float(kline[0][9]),
            float(kline[0][10]),
            int(kline[0][11])
        ]

        self.df.loc[len(self.df.index)] = mod_kline

        # update local csv
        self.df.to_csv(os.path.join(os.getcwd(), 'data', f'{self.price_hist_csv}.csv'), index=False)

        # update Cloud Storage
        blob = self.bucket.blob(self.price_hist_csv)
        with blob.open("w") as f:
            f.write(self.df.to_csv(index=False))

    def make_observation(self):
        step = len(self.df) - 1

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
        print('ok', action)
        if action == 0 and \
                self.balances['USDT'] > self.initial_account_balance * self.kill_thresh and \
                self.balances['BTC'] < 0.00001:
            amount = self.balances['USDT'] * self.percent_capital
            quantity = amount / self.df.loc['_Close'][-1:]
            quantity = math.floor(quantity * (10 ** 6))/ (10 ** 6)
            print(quantity)
            response = self.binance_client.buy_asset('BTCUSDT', quantity)
        elif action == 1 and self.balances['BTC'] > 0.000001:
            quantity = self.balances['BTC']
            quantity = math.floor(quantity * (10 ** 6)) / (10 ** 6)
            print('ok2', quantity)
            response = self.binance_client.sell_asset('BTCUSDT', quantity)
        #time.sleep(2)
        # TODO CHECK STATE OF ORDER
        print("getting state again")
        self.get_binance_state()
        self.datastore_client.create_transaction(self.state['step'],
                                                 'live',
                                                 self.state['session_id'],
                                                 **{"Net Worth (USD)": self.balances['USDT'],
                                                    "Net Worth (BTC)": self.balances['BTC'] * self.df.loc[len(self.df) - 1, '_Close'].item(),
                                                    "Close Price (BTC)": self.df.loc[len(self.df) - 1, '_Close'].item()})
        self.state['step'] = self.state['step'] + 1
        self.save_state()

    def train(self, file_name, interval):
        file_path = os.path.join(os.getcwd(), 'data', f'{file_name}.csv')
        if os.path.exists(file_path):
            data_prepper = RLDataPrepper(interval, file=file_path)
            df = data_prepper.do_it_all()
        else:
            blob = self.bucket.blob(file_name)
            if blob.exists():
                with blob.open("r") as f:
                    df = pd.read_csv(f)
                df.to_csv(file_path, index=False)
                data_prepper = RLDataPrepper(interval, df=df)
                df = data_prepper.do_it_all()
            else:
                raise FileNotFoundError

        env = StockTradingEnv(df, self.initial_account_balance, self.trading_costs_rate,
                              self.max_open_positions, self.percent_capital, self.kill_thresh)

        agent = Agent(n_actions=env.action_space.n, batch_size=self.batch_size,
                      alpha=self.alpha, n_epochs=self.n_epochs,
                      input_dims=env.observation_space.shape,
                      actor_file=f'actor_ppo_{interval}',
                      critic_file=f'critic_ppo_{interval}')

        actor_file_path = os.path.join(os.getcwd(), 'models', f'actor_ppo_{interval}')
        critic_file_path = os.path.join(os.getcwd(), 'models', f'critic_ppo_{interval}')
        if os.path.exists(actor_file_path) and os.path.exists(critic_file_path):
            agent.load_models()

        best_score = env.reward_range[0]
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0
        cached_transactions = []

        dw = DatastoreWrapper()
        session_id = dw.create_session(**{"session_name": "RL train session", "type": "training", "model_name": "RL Model",
                                          "starting_balance": 0.0, "starting_coins": 0.0, "crypto_type": "BTC"})

        print("... starting ...")
        for i in range(self.n_episodes):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                print(f'\r{env.current_step - env.lag}/{env.max_steps - env.lag}', end='')
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                n_steps += 1
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % self.N == 0:
                    agent.learn()
                observation = observation_
            print('\r', end='')

            # Save history
            score_history.append(score)
            avg_score = np.mean(score_history[-50:])

            if avg_score > best_score and i > 9 and avg_score > 0:
                best_score = avg_score
                agent.save_models()
                try:
                    blob = self.bucket.blob(f'actor_ppo_{interval}')
                    blob.upload_from_filename(actor_file_path)
                    blob = self.bucket.blob(f'critic_ppo_{interval}')
                    blob.upload_from_filename(critic_file_path)
                except:
                    print("error uploading models")

            # print(f"episode: {i+1}, score: {score:.2f}, avg score: {avg_score:.2f}, time_steps (current/total): {env.current_step - env.lag}/{n_steps}, learning steps: {learn_iters}")
            # print(f"\ttrades (l/s/h): {env.num_trades_long}/{env.num_trades_short}/{env.num_holds}, profit: {env.net_profit:.2f}, invalid decisions: {env.total_invalid_decisions}")
            print(f"episode: {i}, score: {score}, avg score: {avg_score}, best_score: {best_score if best_score != float('-inf') else 0}")

            # attempt to create transaction in datastore and cache the transaction in case of failure
            new_transaction = {"score": score, "average score": avg_score, "best score": best_score if best_score != float('-inf') else 0}
            try:
                dw.create_transaction(i+1, 'training', session_id, **new_transaction)
            except:
                cached_transactions.insert(0, {i+1: new_transaction})

            # attempt to unload cached transactions
            for i in range(len(cached_transactions)):
                transaction = cached_transactions.pop()
                try:
                    p = [(k, v) for k, v in transaction.items()][0]
                    dw.create_transaction(p[0], 'training', session_id, **p[1])
                except:
                    cached_transactions.insert(0, transaction)


@app.route('/state', methods=['GET'])
def get_state():
    state_file_path = os.path.join(os.getcwd(), 'state.p')
    if os.path.exists(state_file_path):
        with open(state_file_path, "rb") as f:
            state = pickle.load(f)
        return jsonify(state), 200
    return '', 404


@app.route('/balances', methods=['GET'])
def get_balances():
    secret_client = SecretClient()
    binance_secret = secret_client.get_secret("binance-api-secret-hobs")
    binance_key = secret_client.get_secret("binance-api-key-hobs")

    bc = BinanceAPI(binance_key, binance_secret)
    return jsonify(bc.get_account_balances()), 200


@app.route('/infer', methods=['GET'])
def make_decision():
    rl_runner = RL()
    try:
        rl_runner.run()
    except:
        return 'error running...', 500
    else:
        return 'success', 200


@app.route('/train', methods=['POST'])
def train_model():
    data = request.json()
    try:
        file_name = data['file']
        interval = data['interval']
    except KeyError:
        return 'invalid request', 400

    rl_runner = RL()
    rl_runner.train(file_name, interval)
    return 'success', 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7890)  # TODO add SSL/HTTPS support
    #rl_runner = RL()
    #rl_runner.run()
    """
    
    
    rl_runner.train('train_2021-06-01-2021-09-30_1h', '1h')

    secret_client = SecretClient()
    binance_secret = secret_client.get_secret("binance-api-secret-hobs")
    binance_key = secret_client.get_secret("binance-api-key-hobs")

    binance_client = BinanceAPI(binance_key, binance_secret)
    print(binance_client.get_account_balances())
    #print(binance_client.buy_asset('BNBBTC', 90.0))
    print(binance_client.get_account_balances())"""
