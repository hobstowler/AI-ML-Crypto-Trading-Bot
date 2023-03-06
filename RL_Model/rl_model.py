# adapted from: https://www.udemy.com/course/machine-learning-applied-to-stock-crypto-trading-python/

import os
import numpy as np
import matplotlib.pyplot as plt
import torch as T

from RL_Model.classes.agent import Agent
from RL_Model.classes.environment import StockTradingEnv
from data.datastore_wrapper import DatastoreWrapper
from RL_Model.data.rl_data_prep import RLDataPrepper


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
n_episodes = 100
alpha = 0.0003  # learning rate

# ##################################################################
# ###  SCRIPT PARAMETERS
# ##################################################################
train_model = False
test_model = False
train_csv = '../data/train_2021-01-01-2021-12-31_1d.csv'
test_csv = '../data/test_2022-01-01-2022-12-31_1d.csv'


def plot_learning_curve(x, scores, figure_file):
    print("outputting graph...")
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
    plt.plot(x, running_avg)
    plt.plot(x, scores)
    plt.title('Running average of previous 50 scores')
    plt.show()


def train(n_episodes: int, csv: str, interval: str):
    data_prepper = RLDataPrepper(interval, file=csv)
    df = data_prepper.do_it_all()

    env = StockTradingEnv(df, INITIAL_ACCOUNT_BALANCE, TRADING_COSTS_RATE,
                          MAX_OPEN_POSITIONS, PERCENT_CAPITAL, KILL_THRESH)

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
    #session_id = dw.create_session(**{"session_name": "RL train session", "type": "training", "model_name": "RL Model",
    #                                  "starting_balance": 0.0, "starting_coins": 0.0, "crypto_type": "BTC"})

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
            observation = observation_
        print('\r', end='')

        # Save history
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])

        if avg_score > best_score and i > 9:
            best_score = avg_score
            agent.save_models()

        #print(f"episode: {i+1}, score: {score:.2f}, avg score: {avg_score:.2f}, time_steps (current/total): {env.current_step - env.lag}/{n_steps}, learning steps: {learn_iters}")
        #print(f"\ttrades (l/s/h): {env.num_trades_long}/{env.num_trades_short}/{env.num_holds}, profit: {env.net_profit:.2f}, invalid decisions: {env.total_invalid_decisions}")
        print(f"episode: {i}, score: {score}, avg score: {avg_score}, best_score: {best_score}")

        #dw.create_transaction(i+1, 'training', session_id, **{"score": score, "average score": avg_score, "net profit": env.net_profit})

        if (i + 1) % 10 == 0:
            x = [i + 1 for i in range(len(score_history))]
            plot_learning_curve(x, score_history, figure_file)


def report_probabilities(interval: str, csv: str):
    data_prepper = RLDataPrepper(interval, file=csv)
    reporting_df = data_prepper.do_it_all()

    env = StockTradingEnv(reporting_df, INITIAL_ACCOUNT_BALANCE, TRADING_COSTS_RATE,
                          MAX_OPEN_POSITIONS, PERCENT_CAPITAL, KILL_THRESH)

    n_actions = env.action_space.n
    input_dims = env.observation_space.shape

    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=env.observation_space.shape,
                  actor_file=f'actor_ppo_{interval}',
                  critic_file=f'critic_ppo_{interval}')
    if os.path.exists(f'{os.getcwd()}\\models\\actor_ppo_{interval}') and \
            os.path.exists(f'{os.getcwd()}\\models\\critic_ppo_{interval}'):
        agent.load_models()
    model = agent.actor

    long_probs = []
    short_probs = []
    is_long = 1
    is_short = 1
    long_ratio = 0.5
    obs = env.reset()
    for step in range(0, len(reporting_df) - env.lag - 1):
        action, prob, val = agent.choose_action(obs)
        obs, _, _, _ = env.step(action)

        state = T.tensor(obs).float()
        dist = model(state)
        probs = dist.probs.detach().numpy()

        #print(np.argmax(probs), probs)

        long_probs.append(probs[0])
        short_probs.append(probs[1])

    _, p1 = plt.subplots()
    x = [i + 1 for i in range(len(reporting_df) - env.lag - 1)]
    p1.plot(x, long_probs, color='red')
    p1.plot(x, short_probs)
    p2 = p1.twinx()
    p2.plot(x, reporting_df[env.lag + 1:]['_Close'], color='green')
    plt.show()


def test(test_csv, interval):
    data_prepper = RLDataPrepper('1d', file=test_csv)
    df = data_prepper.do_it_all(False)


def plot_prices():
    data_prepper = RLDataPrepper('1d', file=train_csv)
    df = data_prepper.do_it_all(False)
    data_prepper2 = RLDataPrepper('1d', file=test_csv)
    df2 = data_prepper2.do_it_all(False)

    x = [i + 1 for i in range(len(df))]
    x2 = [i + 1 + len(df) for i in range(len(df2))]
    plt.plot(x, df['Close'])
    plt.plot(x, df['bollinger_upper'])
    plt.plot(x, df['bollinger_lower'])
    plt.plot(x2, df2['Close'])
    plt.plot(x2, df2['bollinger_upper'])
    plt.plot(x2, df2['bollinger_lower'])
    plt.show()


if __name__ == '__main__':
    if train_model:
        train(n_episodes, train_csv, '1d')

    if test_model:
        pass

    #plot_prices()
    report_probabilities('1d', train_csv)