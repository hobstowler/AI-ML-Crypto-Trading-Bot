import math

import pandas as pd
import numpy as np
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    """ A stock trading environment with Open AI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, INITIAL_ACCOUNT_BALANCE, TRADING_COSTS_RATE, MAX_OPEN_POSITIONS, PERCENT_CAPITAL, KILL_THRESH):
        super(StockTradingEnv, self).__init__()

        # Generic variables
        self.df = df

        # Account variables
        self.initial_account_balance = INITIAL_ACCOUNT_BALANCE
        self.available_balance = INITIAL_ACCOUNT_BALANCE
        self.profit = 0
        self.net_profit = 0
        self.unrealized_profit = 0

        # Position variables
        self.open_positions = {}
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.num_holds = 0
        self.long_short_ratio = 0
        self.invalid_decisions = 0
        self.total_invalid_decisions = 0
        self.held_for = 0
        self.percent_capital = PERCENT_CAPITAL
        self.trading_cost = TRADING_COSTS_RATE
        self.max_positions = MAX_OPEN_POSITIONS

        # Current Step
        self.current_step = self.lag = 20
        self.volatility = 1
        self.max_steps = len(df)

        # Actions of the format Buy, Sell, Hold
        self.action_space = spaces.Discrete(2)

        # Prices contains the Close and Close Returns etc
        self.observation_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)

    # Reset Environment
    def reset(self):
        self.available_balance = self.initial_account_balance
        self.profit = 0
        self.net_profit = 0
        self.unrealized_profit = 0
        self.open_positions = {}
        self.num_trades_long = 0
        self.num_trades_short = 0
        self.num_holds = 0
        self.invalid_decisions = 0
        self.total_invalid_decisions = 0
        self.held_for = 0
        self.current_step = self.lag
        self.volatility = 1

        return self._next_observation()

    # Reward Structure
    def _calculate_reward(self) -> int:
        reward = 0
        reward += self.net_profit / self.volatility
        reward += 0.01 if self.long_short_ratio >= 0.3 and self.long_short_ratio <= 0.6 else -0.01
        return reward

    # Observation Structure
    def _next_observation(self):
        close = self.df.loc[self.current_step, 'Close'].item()
        volume = self.df.loc[self.current_step, 'Volume'].item()
        sma30 = self.df.loc[self.current_step, 'SMA30'].item()
        ema30 = self.df.loc[self.current_step, 'EMA30'].item()
        cma = self.df.loc[self.current_step, 'CMA'].item()
        bollinger_upper = self.df.loc[self.current_step, 'bollinger_upper'].item()
        bollinger_lower = self.df.loc[self.current_step, 'bollinger_lower'].item()
        macd = self.df.loc[self.current_step, 'macd'].item()
        signal = self.df.loc[self.current_step, 'signal'].item()
        rs = self.df.loc[self.current_step, 'rs'].item()

        env_4 = 1 if self.long_short_ratio else 0

        """open_num = len(self.open_positions)
        open_val = sum([x[1] for x in self.open_positions.values()])

        #unrealized_profit = 0
        if open_num > 0:
            cur_price = self.df.loc[self.current_step, '_Close'].item()
            position_value = sum([x[1] for x in self.open_positions.values()])
            cur_quantity = sum([x[0] for x in self.open_positions.values()])
            cur_value = cur_quantity * (1 - self.trading_cost) * cur_price
            self.unrealized_profit = (cur_value - position_value)"""

        obs = np.array([close, volume, sma30, ema30, cma, bollinger_lower,
                        bollinger_upper, macd, signal, rs, env_4])

        return obs

    # Action Management
    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "_Close"].item()
        next_price = self.df.loc[self.current_step + 1, "_Close"].item()
        next_return = next_price / current_price - 1

        # Go Long
        if action == 0:
            self.net_profit += self.available_balance * self.percent_capital * next_return
            self.available_balance += self.net_profit
            self.num_trades_long += 1

        # Go Short
        if action == 1:
            self.net_profit += self.available_balance * self.percent_capital * -next_return
            self.available_balance += self.net_profit
            self.num_trades_short += 1

        # Update metrics
        self.long_short_ratio = self.num_trades_long / (self.num_trades_long + self.num_trades_short)
        self.volatility = self.df.loc[self.current_step - self.lag, "_Close"].sum()

    # Step Function
    def step(self, action):
        self._take_action(action)

        reward = self._calculate_reward()

        self.current_step += 1

        is_max_steps_taken = self.current_step >= self.max_steps - self.lag - 1
        done = True if is_max_steps_taken else False

        obs = self._next_observation()

        return obs, reward, done, {}

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        pass