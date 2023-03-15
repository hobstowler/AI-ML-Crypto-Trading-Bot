# Description: LSTM model for time series prediction on bitcoin data. Contains functions for model training, 
#              prediction, and visualization.

# Install dependencies
# pip install torch
# pip install matplotlib
# pip install numpy
# pip install pandas (Only needed if training from csv file)

import torch
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd

from lstm.model_tools import *
 
# Add parent directory to path so we can import from data_conversion
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from data_conversion.generate_datasets import tensors_from_csv


# Model hyperparameters
HIDDEN_SIZE = 40
NUM_LAYERS = 2
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 10

# Data hyperparameters
FEATURES = ['close_price', 'volume']
INPUT_SIZE = len(FEATURES)
OUTPUT_SIZE = len(FEATURES)

# For data collection and analysis
MODEL_SAVE_POINTS = [10, 100, 500]
loss_values = []


class LSTM(nn.Module):
    """LSTM neural network architecture.
    """
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1, batch_size=1):
        super(LSTM, self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers
        self.batch_size=batch_size

        # Define the LSTM basic architecture
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Linear layer connecting hidden layer to output ("top" of the LSTM)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # Initialize hidden layer as zeros
    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size), 
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    # Defines what happens when the model is run. Basically the base LSTM runs, then the linear layer for output runs
    def forward(self, x, hidden_prev):
        out, hidden_prev = self.lstm(x, hidden_prev)
        out = self.linear(out)
        return out, hidden_prev

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


def train_loop(save_points, model_name, hyperparams, data_source_info, plot=False):
    """
    Train with the specified hyperparameters over the given number of epochs
    """

    input_size = len(data_source_info['columns'])
    output_size = input_size

    # Unpack hyperparameters
    hidden_size = hyperparams['hidden_size']
    num_layers = hyperparams['num_layers']
    batch_size = hyperparams['batch_size']
    epochs = hyperparams['epochs']
    lr = hyperparams['lr']

    # Unpack data source info
    train_source = data_source_info['train_source']
    val_source = data_source_info['val_source']
    test_source = data_source_info['test_source']
    seq_len = data_source_info['seq_len']
    columns = data_source_info['columns']
    
    # Initialize model
    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                num_layers=num_layers, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Get data in tensor form
    train_tensors = tensors_from_csv(train_source, seq_len=seq_len, columns=columns, batch_size=batch_size)
    val_tensors = tensors_from_csv(val_source, seq_len=seq_len, columns=columns, batch_size=batch_size)
    test_tensors = tensors_from_csv(test_source, seq_len=seq_len, columns=columns, batch_size=1)

    train_losses = []
    val_losses = []
    pred_losses = []

    # Perform training over each epoch
    for epoch in range(epochs):
        train_loss = train_lstm(model, criterion, train_tensors, optimizer, save_points=save_points)
        val_loss = validate_lstm(model, criterion, val_tensors)
        _, pred_loss = predict(model, criterion, test_tensors, pred_len=6)

        print(f'Epoch {epoch}: training loss {train_loss} | val loss {val_loss} | pred loss {pred_loss}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        pred_losses.append(pred_loss)


    if plot:
        # First show the training and validation loss over each epoch
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(train_losses, label='training')
        plt.plot(val_losses, label='validation')
        plt.plot(pred_losses, label='prediction')
        plt.legend()
        plt.show()
        
        predictions, pred_loss = predict(model, criterion, test_tensors, pred_len=6)
        vizualize_predictions(predictions, test_tensors)

    save_model(model, optimizer, f'./models/model_{model_name}.pt')

    results = {}
    results['train_loss'] = train_losses
    results['val_loss'] = val_losses
    results['test_loss'] = pred_losses

    return results

def train_from_csv(infile, outfile, features = FEATURES):
    """
    Loop through a csv file and perform training on each set of hyperparameters
    """

    # Read in csv file
    df = pd.read_csv(infile)

    # Make list of epochs from max value in csv
    epochs = list(range(1, int(df['epochs'].max()) + 1))

    # Make the column list
    columns = ['type'] + df.columns.values.tolist() + epochs

    # Make output dataframe
    out_df = pd.DataFrame(columns=columns)

    # For each set of supplied hyperparameters perform training
    for idx in range(len(df)):

        print(f'Hyperparameters: {df.loc[idx]}')

        hyperparams = {}
        # Unpack hyperparameters
        hyperparams['hidden_size'] = df.loc[idx, 'hidden_size']
        hyperparams['num_layers'] = df.loc[idx, 'num_layers']
        hyperparams['batch_size'] = df.loc[idx, 'batch_size']
        hyperparams['epochs'] = df.loc[idx, 'epochs']
        hyperparams['lr'] = df.loc[idx, 'lr']
    
        # Set data source
        data_source_info = {
            'train_source': './train_48.csv',
            'val_source': './val_48.csv',
            'test_source': './test_48.csv',
            'seq_len': 48,
            'columns': features
        }
    
        results = train_loop(MODEL_SAVE_POINTS, 'test_1', hyperparams, data_source_info, plot=True)
        
        types = ['train', 'val', 'test']

        for i in range(len(types)):
            out_df.loc[idx * 3 + i, 'type'] = types[i]
            for key, value in hyperparams.items():
                out_df.loc[idx * 3 + i, key] = value

            for epoch, loss in enumerate(results[f'{types[i]}_loss']):
                out_df.loc[idx * 3 + i, epoch + 1] = loss

    # Save output dataframe to csv
    out_df.to_csv(outfile)



if __name__=='__main__':

    # Initialize hyperparameters and data source and perform one training loop

    # hyperparams = {
    #     'input_size': INPUT_SIZE,
    #     'hidden_size': HIDDEN_SIZE,
    #     'output_size': OUTPUT_SIZE,
    #     'num_layers': NUM_LAYERS,
    #     'batch_size': BATCH_SIZE,
    #     'epochs': EPOCHS,
    #     'lr': LR
    # }

    # data_source_info = {
    #     'train_source': './train_48.csv',
    #     'val_source': './val_48.csv',
    #     'test_source': './test_48.csv',
    #     'seq_len': 48,
    #     'columns': ['close_price']
    # }

    # train_loop(MODEL_SAVE_POINTS, 'test_1', hyperparams, data_source_info, plot=True)

    train_from_csv(infile='./hyperparams.csv', outfile='./results.csv')







    


