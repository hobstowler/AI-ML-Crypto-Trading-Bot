
# pip install torch

import torch
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import sys
import numpy as np

from model_tools import save_model
 
# Add parent directory to path so we can import from data_conversion
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from data_conversion.generate_datasets import tensors_from_csv


# For building sine data
NUM_TIME_STEPS = 48
CONTEXT_LENGTH = 12
NUM_CYCLES = 6

# Model hyperparameters
HIDDEN_SIZE = 10
INPUT_SIZE = 1
OUTPUT_SIZE = 1
NUM_LAYERS = 1
BATCH_SIZE = 2
LR = 0.001
EPOCHS = 1

# For data collection and analysis
MODEL_SAVE_POINTS = [10, 100, 500]
loss_values = []


class Net(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1, num_layers=1, batch_size=1):
        super(Net, self).__init__()

        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers
        self.batch_size=batch_size

        # Define our LSTM basic architecture
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

    # Defines what happens when the model is run. Basically the base RNN runs, then the linear layer for output runs
    def forward(self, x, hidden_prev):
        out, hidden_prev = self.lstm(x, hidden_prev)
        out = self.linear(out)
        return out, hidden_prev

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


def train(model, criterion, data_tensors, optimizer, save_points):
    # Initialize hidden layer to zeros
    hidden_prev = model.init_hidden()

    iter = 0
    for curr_tensor in data_tensors:

        curr_tensor = curr_tensor.to(torch.float32)

        # Split into input and target values
        inputs, targets = curr_tensor[:,:-1,:], curr_tensor[:,1:,:]
        output, hidden_prev = model(inputs, hidden_prev)
        hidden_prev = (hidden_prev[0].detach(), hidden_prev[1].detach())
        loss = criterion(output, targets)
        model.zero_grad()
        loss.backward()

        # Update weights based on mainly the new gradients and learning rate
        optimizer.step()

        loss_values.append(loss.item())

        if iter % 1 == 0:
            print(f'iteration: {iter}, training loss {loss.item()}')

        # Save a model from each checkpoint
        if iter in save_points:
            save_model(model, optimizer, f'./models/model_{iter}.pt')

        iter += 1

    plt.plot(loss_values)
    plt.show()

def validate(model, criterion, data_tensors):
    # Initialize hidden layer to zeros
    hidden_prev = model.init_hidden()

    iter = 0
    with torch.no_grad():
        for curr_tensor in data_tensors:

            curr_tensor = curr_tensor.to(torch.float32)

            # Split into input and target values
            inputs, targets = curr_tensor[:,:-1,:], curr_tensor[:,1:,:]
            output, hidden_prev = model(inputs, hidden_prev)
            hidden_prev = (hidden_prev[0].detach(), hidden_prev[1].detach())
            loss = criterion(output, targets)

            loss_values.append(loss.item())

            if iter % 1 == 0:
                print(f'iteration: {iter}, validation loss {loss.item()}')

            iter += 1

def predict(model, criterion, data_tensors, pred_len):
    # Initialize hidden layer to zeros

    model.set_batch_size(1)

    iter = 0
    predictions = []
    with torch.no_grad():
        for curr_tensor in data_tensors:
            hidden_prev = model.init_hidden()

            curr_tensor = curr_tensor.to(torch.float32)

            split_point = len(curr_tensor[0]) - pred_len

            # Split into input and target values
            inputs, targets = curr_tensor[:,:split_point,:], curr_tensor[:,split_point:,:]
            # Run through the inputs to build up the hidden state
            output, hidden_prev = model(inputs, hidden_prev)
            
            prediction = torch.zeros(targets.shape)
            # Make predictions for each value in the target
            for i in range(pred_len):
                # Use the last output as the next input
                output, hidden_prev = model(output[:,-1,:].unsqueeze(1), hidden_prev)
                prediction[:,i,:] = output

            loss = criterion(prediction, targets)

            loss_values.append(loss.item())

            if iter % 1 == 0:
                print(f'iteration: {iter}, prediction loss {loss.item()}')

            iter += 1

            predictions.append(prediction)

    return predictions

def vizualize_predictions(predictions, targets, index=0):

    grid_rows = int(np.ceil(np.sqrt(len(predictions))))
    grid_cols = int(np.ceil(len(predictions)/grid_rows))
    

    figure, axis = plt.subplots(grid_rows, grid_cols)

    for row in range(grid_rows):
        for col in range(grid_cols):
            index = row*grid_cols + col
            if index < len(predictions):
                x_targets = np.arange(0,len(targets[0].squeeze().numpy()))
                start_point = len(targets[0].squeeze().numpy()) - len(predictions[index].squeeze().numpy())
                x_predictions = np.arange(start_point, len(x_targets))

                axis[row][col].scatter(x_targets, targets[index].squeeze().numpy())
                axis[row][col].scatter(x_predictions, predictions[index].squeeze().numpy())
                
    plt.savefig('predictions.png')

    plt.show()



def train_loop(save_points, model_name, hyperparams, data_source_info):

    # Unpack hyperparameters
    input_size = hyperparams['input_size']
    hidden_size = hyperparams['hidden_size']
    output_size = hyperparams['output_size']
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
    model = Net(input_size=input_size, hidden_size=hidden_size, output_size=output_size, 
                num_layers=num_layers, batch_size=batch_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    # Get data in tensor form
    train_tensors = tensors_from_csv(train_source, seq_len=seq_len, columns=columns, batch_size=batch_size)
    val_tensors = tensors_from_csv(val_source, seq_len=seq_len, columns=columns, batch_size=batch_size)
    test_tensors = tensors_from_csv(test_source, seq_len=seq_len, columns=columns, batch_size=1)

    # Perform training over each epoch
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        train(model, criterion, train_tensors, optimizer, save_points=save_points)
        validate(model, criterion, val_tensors)

    # Make predictions on test data
    predictions = predict(model, criterion, test_tensors, 12)
    vizualize_predictions(predictions, test_tensors, 7)

    save_model(model, optimizer, f'./models/model_{model_name}.pt')

if __name__=='__main__':

    hyperparams = {
        'input_size': INPUT_SIZE,
        'hidden_size': HIDDEN_SIZE,
        'output_size': OUTPUT_SIZE,
        'num_layers': NUM_LAYERS,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'lr': LR
    }

    data_source_info = {
        'train_source': './train_48.csv',
        'val_source': './val_48.csv',
        'test_source': './test_48.csv',
        'seq_len': 48,
        'columns': ['close_price']
    }

    train_loop(MODEL_SAVE_POINTS, 'test_1', hyperparams, data_source_info)






    


