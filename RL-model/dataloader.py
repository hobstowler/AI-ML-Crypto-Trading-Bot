import torch
import pandas as pd
import os
from torch.utils.data import Dataset
import matplotlib as plt

class CustomCryptoDataset(Dataset):
    def __init__(self, trade_data_file,):
        self.trade_data = pd.read_csv(trade_data_file)

    def __len__(self):
        return len(self.trade_data)

    def __getitem__(self, idx):
        pass