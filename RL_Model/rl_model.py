import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import deque, namedtuple
import pandas as pd
import numpy as np
import random
import os

steps_taken = 0

training = True
TRAIN_DATA_PATH = f'{os.getcwd()}/training_data.csv'
TEST_DATA_PATH = f'{os.getcwd()}/testing_data.csv'
print(f'Using training data at: {TRAIN_DATA_PATH if training else TEST_DATA_PATH}')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} device.')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

''' ################################################
#   HYPERPARAMETERS                                #
################################################ '''
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.79
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

class ReplayMemory(object):
    def __init__(self, lookback=100):
        self.memory = deque([], maxlen=lookback)

    def push(self, *args):
        self.memory.append(*args)  # TODO

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def sample(csv=True) -> float:
    global steps_taken

    if csv:
        pass
    else:
        # Make call into Binance API
        pass


def buy_crypto(price, amount):
    pass


def sell_crypto(price, amount):
    pass


def import_data():
    if training:
        return pd.read_csv(TRAIN_DATA_PATH)
    else:
        return pd.read_csv(TEST_DATA_PATH)


dp = [0]


def max_profit() -> int:
    """
    Part of loss function.
    :param next_price:
    :return:
    """
    global steps_taken
    global prices
    global dp  # TODO tensor-ize it
    if steps_taken == 0:
        return 0
    dp.append(dp[steps_taken - 1] + max(0, prices[steps_taken] - prices[steps_taken - 1]))

    return dp[steps_taken]


if __name__ == '__main__':
    global steps_taken
    num_episodes = 600 if torch.cuda.is_available() else 300

    # get data for testing/training
    data = import_data()

    for episode in range(num_episodes):
        # reset state



        # iterate through training/test data
        for t in len(data):
            pass