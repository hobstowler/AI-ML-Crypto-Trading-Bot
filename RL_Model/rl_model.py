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

from RL_Model.classes.agent import Agent
from RL_Model.classes.environment import StockTradingEnv
from RL_Model.classes.ppo_memory import PPOMemory
from RL_Model.classes.actor import ActorNetwork
from RL_Model.classes.critic import CriticNetwork
from data.datastore_wrapper import DatastoreWrapper
from rl_data_prep import RLDataPrepper


# ##################################################################
# ###  TRADING PARAMETERS
# ##################################################################
MAX_OPEN_POSITIONS = 1
INITIAL_ACCOUNT_BALANCE = 1000
PERCENT_CAPITAL = 0.1
TRADING_COSTS_RATE = 0.001  # cost to execute a trade. will be multiplied by the number of open positions
KILL_THRESH = 0.4  # terminate if balance too low. Acts as a percentage of initial net worth

# ##################################################################
# ###  HYPER PARAMETERS
# ##################################################################
N = 20
batch_size = 10
n_epochs = 5
n_episodes = 10
alpha = 0.0003  # learning rate

# ##################################################################
# ###  SCRIPT PARAMETERS
# ##################################################################
train_model = True
test_model = False
train_csv = '../data/all_2021-01-01-2021-12-31_1h.csv'
test_csv = '../data/all_2021-01-01-2022-12-31_1h.csv'


def plot_learning_curve(x, scores, figure_file):
    print("outputting graph...")
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.plot(x, scores)
    plt.title('Running average of previous 50 scores')
    plt.show()


def train_model(n_episodes: int, csv: str, interval: str):
    data_prepper = RLDataPrepper(csv, interval)
    df = data_prepper.do_it_all()

    env = StockTradingEnv(df, INITIAL_ACCOUNT_BALANCE, TRADING_COSTS_RATE,
                          MAX_OPEN_POSITIONS, PERCENT_CAPITAL, KILL_THRESH)
    N = 20
    batch_size = 5
    n_epochs = 3
    alpha = 0.0003
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  actor_file=f'actor_ppo_{interval}',
                  critic_file=f'critic_ppo_{interval}')

    if os.path.exists(f'{os.getcwd()}\\models\\actor_ppo_{interval}') and \
            os.path.exists(f'{os.getcwd()}\\models\\critic_ppo_{interval}'):
        agent.load_models()

    figure_file = f'stock_training_{interval}.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    dw = DatastoreWrapper()
    session_id = dw.create_session(**{"session_name": "RL train session", "type": "training", "model_name": "RL Model",
                                      "starting_balance": 0.0, "starting_coins": 0.0, "crypto_type": "BTC"})

    print("... starting ...")
    for i in range(n_episodes):
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
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        print('\r', end='')

        # Save history
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score and i > 9:
            best_score = avg_score
            agent.save_models()

        print(f"episode: {i+1}, score: {score:.2f}, avg score: {avg_score:.2f}, time_steps (current/total): {env.current_step - env.lag}/{n_steps}, learning steps: {learn_iters}")
        print(f"\ttrades (l/s): {env.num_trades_long}/{env.num_trades_short}, profit: {env.net_profit:.2f}, invalid decisions: {env.total_invalid_decisions}")

        dw.create_transaction(i+1, 'training', session_id, **{"score": score, "average score": avg_score, "net profit": env.net_profit})

        if (i + 1) % 10 == 0:
            x = [i + 1 for i in range(len(score_history))]
            plot_learning_curve(x, score_history, figure_file)


    def test_model(test_csv, interval):
        data_prepper = RLDataPrepper(test_csv, '1h')
        df = data_prepper.do_it_all(False)


if __name__ == '__main__':
    if train_model:
        train_model(n_episodes, train_csv, '1h')

    if test_model:
        pass
    """
    data_prepper = RLDataPrepper(test_csv, '1h')
    df = data_prepper.do_it_all(False)
    x = [x + 1 for x in range(len(df))]
    plt.plot(x, df['Close'])
    plt.plot(x, df['bollinger_upper'])
    plt.plot(x, df['bollinger_lower'])
    plt.show()
    data_prepper = RLDataPrepper('../data/all_2021-01-01-2022-12-31_1h.csv', '1h')
    df = data_prepper.do_it_all()

    env = StockTradingEnv(df, INITIAL_ACCOUNT_BALANCE, TRADING_COSTS_RATE,
                          MAX_OPEN_POSITIONS, PERCENT_CAPITAL, KILL_THRESH)
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, alpha=alpha,
                  n_epochs=n_epochs, input_dims=env.observation_space.shape)

    if os.path.exists(f'{os.getcwd()}\\tmp\\actor_torch_ppo_sine_new') and \
        os.path.exists(f'{os.getcwd()}\\tmp\\critic_torch_ppo_sine_new'):
        agent.load_models()

    best_score = env.reward_range[0]
    trade_history = []
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
            print(f'\r{e_steps}/{env.max_steps}', end='')
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
        print('\r', end='')

        # Save history
        trade_history.append(env.num_trades)
        score_history.append(score)
        profit_history.append(env.unrealized_profit + env.realized_profit)
        net_worth_history.append(env.net_worth)
        step_history.append(e_steps)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score and i > 50:
            best_score = avg_score
            agent.save_models()

        print(f"episode: {i+1}, score: {score:.2f}, avg score: {avg_score:.2f}, time_steps (current/total): {e_steps}/{n_steps}, learning steps: {learn_iters}")

        x = [i+1 for i in range(len(score_history))]
        plt.plot(x, score_history, label='score')
        plt.plot(x, step_history, label='steps')
        plt.plot(x, profit_history, label='profit')
        plt.plot(x, net_worth_history, label='net worth')
        plt.plot(x, trade_history, label='number of trades')
        plt.yscale("linear")
        plt.legend(loc='best')
        plt.show()"""

