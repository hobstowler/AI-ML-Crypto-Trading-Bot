# adapted from: https://www.udemy.com/course/machine-learning-applied-to-stock-crypto-trading-python/

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import gym
from gym import spaces
import random


# ##################################################################
# ###  TRADING PARAMETERS
# ##################################################################
MAX_OPEN_POSITIONS = 3
INITIAL_ACCOUNT_BALANCE = 1000
PERCENT_CAPITAL = 0.05
TRADING_COSTS_RATE = 0.001  # cost to execute a trade. will be multiplied by the number of open positions
KILL_THRESH = 0.4  # terminate if balance too low. Acts as a percentage of initial net worth

# ##################################################################
# ###  HYPER PARAMETERS
# ##################################################################
N = 20
batch_size = 5
n_epochs = 10
n_episodes = 200
alpha = 0.0003  # learning rate


# Create timesteps
time = np.arange(0, 50, 0.1)

# Assign Amplitude and normalize above 0
amplitude = np.sin(time)
amplitude = amplitude + 1
max_amp = max(amplitude)
amplitude = amplitude / max_amp

# Construct DataFrame
df = pd.DataFrame(amplitude)
df.columns = ['Close']
df['Close_Rt'] = df['Close'].pct_change()
df = df.replace(np.inf, np.nan)
df = df.dropna()
df = df.reset_index(drop=True)


# Build Environment Class
class StockTradingEnv(gym.Env):
    """ A stock trading environment with Open AI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame):
        super(StockTradingEnv, self).__init__()

        # Generic Variables
        self.df = df

        # Account Variables
        self.account_balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.last_profit = 0

        # Position Variables
        self.open_quantities = []
        self.open_prices = []
        self.trading_costs = 0
        self.open_positions = 0
        self.closed_positions = 0
        self.incorrect_position_calls = 0
        self.num_trades = 0
        self.held_for_period = 0

        # Current Step
        self.current_step = 0
        self.max_steps = len(df)

        # Actions of the format Long, Hold, Close
        self.action_space = spaces.Discrete(3)

        # Set up observation space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

    # Reward Structure
    def _calculate_reward(self):
        reward = 0
        if self.num_trades:
            reward += self.realized_profit / self.num_trades
            reward += self.unrealized_profit / self.num_trades * 0.3
            reward += 1 if self.last_profit > 0 else 0
        reward -= 1 if self.incorrect_position_calls > 0 else 0
        if reward <= 0:
            reward -= 2  # incentivize trading over holding
        return reward

    # Observation Structure
    def _next_observation(self):
        close_item = self.df.loc[self.current_step, 'Close'].item()
        close_rt_item = self.df.loc[self.current_step, 'Close_Rt'].item()
        close_T1_item = self.df.loc[self.current_step - 1, 'Close_Rt'].item()
        close_T2_item = self.df.loc[self.current_step - 2, 'Close_Rt'].item()
        close_T3_item = self.df.loc[self.current_step - 3, 'Close_Rt'].item()
        close_T4_item = self.df.loc[self.current_step - 4, 'Close_Rt'].item()

        current_position = 1 if self.open_positions else 0
        num_trades = self.num_trades / len(self.df) if self.num_trades > 0 else 0

        obs = np.array([close_item, close_rt_item, close_T1_item, close_T2_item,
                        close_T3_item, close_T4_item, current_position, num_trades])

        return obs

    # Calculate open positions value
    def _calculate_open_value(self):
        open_trades_value = 0
        counts = 0
        for qty in self.open_quantities:
            acquisition_price = self.open_prices[counts]
            open_trades_value += acquisition_price * qty
            counts += 1
        return open_trades_value

    # Calculate net profit
    def _profit_calculation(self, current_price, calc_type):
        open_trades_value = self._calculate_open_value()
        total_quantity_held = sum(self.open_quantities)
        current_value = total_quantity_held * current_price
        gross_profit = current_value - open_trades_value

        if calc_type == 'close_position':
            trading_costs = current_value * TRADING_COSTS_RATE
            self.trading_costs += trading_costs
        elif calc_type == 'hold_position' or calc_type == 'open_position':
            trading_costs = open_trades_value * TRADING_COSTS_RATE

        net_profit = gross_profit - trading_costs

        return net_profit

    # Action Management
    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "Close"].item()

        # Reset last profit
        self.last_profit = 0
        self.incorrect_position_calls = 0

        # Go Long
        if action == 0:
            if self.open_positions < MAX_OPEN_POSITIONS:
                net_profit = self._profit_calculation(current_price, "open_position")
                net_worth = self.net_worth + net_profit
                trading_allowance = net_worth * PERCENT_CAPITAL

                self.open_quantities.append(trading_allowance / current_price)
                self.open_prices.append(current_price)
                self.trading_costs += trading_allowance * TRADING_COSTS_RATE
                self.num_trades += 1
            else:
                self.incorrect_position_calls += 1

        # Hold Positions
        if action == 1:
            net_profit = self._profit_calculation(current_price, "hold_position")
            self.unrealized_profit += net_profit
            if self.open_positions > 0:
                self.held_for_period += 1

        # Close Positions
        if action == 2:
            if self.open_positions != 0:
                net_profit = self._profit_calculation(current_price, "close_position")
                self.last_profit = net_profit
                self.realized_profit += net_profit
                self.unrealized_profit = 0
                self.open_quantities = []
                self.open_prices = []
                self.held_for_period = 0
                self.closed_positions += 1
            else:
                self.incorrect_position_calls += 1

        # Update variables
        open_trades_value = self._calculate_open_value()
        self.open_positions = len(self.open_quantities)
        self.net_worth = INITIAL_ACCOUNT_BALANCE + self.unrealized_profit + self.realized_profit
        self.available_balance = INITIAL_ACCOUNT_BALANCE - open_trades_value + self.realized_profit

    # Step Function
    def step(self, action):
        self._take_action(action)

        reward = self._calculate_reward()

        self.current_step += 1

        is_max_steps_taken = self.current_step >= self.max_steps - 1
        is_account_balance_reached = self.net_worth <= INITIAL_ACCOUNT_BALANCE * KILL_THRESH
        done = True if is_max_steps_taken or is_account_balance_reached else False

        obs = self._next_observation()

        return obs, reward, done, {}

    # Reset Environment
    def reset(self):
        self.account_balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.open_quantities = []
        self.open_prices = []
        self.trading_costs = 0
        self.open_positions = 0
        self.incorrect_position_calls = 0
        self.current_step = 5

        return self._next_observation()

    # Render the environment to the screen
    def render(self, mode='human', close=False):
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        return profit


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo_sine')
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)

        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo_sine')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.AdamW(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda

        self.actor = ActorNetwork(n_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(np.array([observation]), dtype=T.float).to(self.actor.device)
        state = state.flatten(0)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, \
            dones_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 + critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()


def load_model(type: str, path: str) -> nn.Module:
    model = ActorNetwork()
    return model


if __name__ == '__main__':


    env = StockTradingEnv(df)
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                  n_epochs=n_epochs, input_dims=env.observation_space.shape)

    if os.path.exists(f'{os.getcwd()}\\tmp\\actor_torch_ppo_sine') and \
        os.path.exists(f'{os.getcwd()}\\tmp\\critic_torch_ppo_sine'):
        agent.load_models()

    figure_file = "sinewave.png"
    best_score = env.reward_range[0]
    score_history = []
    profit_history = []
    net_worth_history = []
    step_history = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0

    print('...starting...')
    for i in range(n_episodes):
        observation = env.reset()
        done = False
        score = 0
        e_steps = 0
        while not done:
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1
            e_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_

        # Save history
        score_history.append(score)
        profit_history.append(env.unrealized_profit + env.realized_profit)
        net_worth_history.append(env.net_worth)
        step_history.append(e_steps)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score and i > 10:
            best_score = avg_score
            agent.save_models()

        print(f"episode: {i}, score: {score:.2f}, avg score: {avg_score:.2f}, time_steps (current/total): {e_steps}/{n_steps}, learning steps: {learn_iters}")

    x = [i+1 for i in range(len(score_history))]
    plt.plot(x, score_history, label='score')
    plt.plot(x, step_history, label='steps')
    plt.plot(x, profit_history, label='profit')
    plt.plot(x, net_worth_history, label='net worth')
    plt.yscale("linear")
    plt.legend(loc='best')
    plt.show()

