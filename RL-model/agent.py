import random

import torch as tf
import numpy as np
import pandas as pd
from collections import deque


class CryptoAgent:
    def __init__(self, data: pd.DataFrame, starting_balance: int=10, lookback_size: int = 50):
        self.data = data
        self.total_steps = len(self.data) - 1
        self.starting_balance = starting_balance
        self.lookback_size = lookback_size

        self.positions = []

        self.action_space = tf.tensor([0, 1, 2])  # buy, sell, hold
        self.order_history = deque(maxlen=lookback_size)
        self.market_history = deque(maxlen=lookback_size)
        self.state_size = (self.lookback_size, 10)

    def step(self, action):
        pass

    def reset(self, env_steps_size=0):
        self.balance = self.net_worth = self.prev_net_worth = self.initial_balance
        self.crypto_held = self.crypto_sold = self.crypto_bought = 0

        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(self.lookback_size, self.total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_size
            self.end_step = self.total_steps

        self.current_step = self.start_step

    def render(self):
        pass

    def _next_observation(self):
        self.market_history.append()
        return np.concatenate((self.market_history, self.order_history), axis=1)

