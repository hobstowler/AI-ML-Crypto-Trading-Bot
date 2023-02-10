
# pip install torch

import torch
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import sys
 
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
BATCH_SIZE = 4
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
        out = out.unsqueeze(dim=0)
        return out, hidden_prev

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
            print(f'iteration: {iter}, loss {loss.item()}')

        # Save a model from each checkpoint
        if iter in save_points:
            # check if the models folder exists and if not make it
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(), f'./models/model_{iter}.pt')

        iter += 1

    plt.plot(loss_values)
    plt.show()


if __name__=='__main__':
    model = Net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_LAYERS, BATCH_SIZE)
    criterion = nn.MSELoss()                        # Set the loss criteria as mean squared error
    optimizer = optim.Adam(model.parameters(), LR)  # This will facilitate gradient updates, Adam is a very common method

    data_tensors = tensors_from_csv('./train_48.csv', seq_len=48, columns=['close_price'], batch_size=BATCH_SIZE)

    train(model, criterion, data_tensors, optimizer, MODEL_SAVE_POINTS)






    


