# Adapted from: https://medium.com/dejunhuang/learning-day-27-implementing-rnn-in-pytorch-for-time-series-prediction-3ddb6190e83d

# IMPORTS

import random
import torch
from torch import nn, optim
import os
import sys
import matplotlib.pyplot as plt

# data handling
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data_conversion.generate_datasets import tensors_from_csv
from data_conversion.generate_datasets import generate_csv_datasets
from data_conversion.generate_datasets import clean_dataset_csv_files


# CONSTANTS

# Model hyperparameters
HIDDEN_SIZE = 200
INPUT_SIZE = 10
OUTPUT_SIZE = 10
NUM_LAYERS = 1
LR = 1e-5
EPOCHS = 50


class Net(nn.Module):
    
    def __init__(
        self, input_size, hidden_size, 
        output_size, num_layers, batch_size
    ):
        super(Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        # Define our RNN basic architecture
        self.rnn = nn.RNN(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True
        )

        # Linear layer connecting hidden layer to output ("top" of the RNN)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # Initialize hidden layer as zeros
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.num_layers, self.hidden_size)

    # Defines what happens when the model is run. Basically the base RNN runs, 
    # then the linear layer for output runs
    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = out.reshape(-1, self.hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev

def main():
    
    # for data saving
    loss_values = []

    # build rnn model
    model = Net(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        output_size=OUTPUT_SIZE, 
        num_layers=NUM_LAYERS,
        batch_size=1
    ) 
    
    i = 0
    for epoch in range(EPOCHS):

        in_len = random.randint(10, 100)
        target_len = random.randint(10, 100)
    
        generate_csv_datasets("training_data/2021.csv", input_len=in_len, target_len=target_len)
        dataset = tensors_from_csv(
            f"data_conversion/train_input_{in_len}.csv", 
            seq_len=48, 
            columns=[
                "open_price", "high_price", "low_price", "close_price", "volume", 
                "close_time", "quote_asset_volume","qty_transactions", 
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
            ], 
            batch_size=1
        )

        # Set the loss criteria as mean squared error
        criterion = nn.MSELoss()

        # This will facilitate gradient updates, Adam is a very common method
        optimizer = optim.Adam(model.parameters(), LR)

        # Initialize hidden layer to zeros
        hidden_prev = model.init_hidden()

        # Perform training over specified number of iterations
        for tensor in dataset:
            
            tensor = tensor.to(torch.float32)
            
            # Split into input and target values
            inputs, targets = tensor[:,:-1,:], tensor[:,1:,:]

            # Run model forward and get predicted values
            output, hidden_prev = model(inputs, hidden_prev)

            # Save the hidden layer as it will be an input for next iteration 
            # (detach removes the new copy from the gradient update graph)
            hidden_prev = hidden_prev.detach()

            # Calculate the loss from current output compared to the target values
            loss = criterion(output, targets)

            # PyTorch thing, need to zero out the calculated gradients from the 
            # previous iteration, otherwise they accumulate
            model.zero_grad()

            # Run backpropogation and calculate new gradients
            loss.backward()

            # Update weights based on mainly the new gradients and learning rate
            optimizer.step()

            #########################################################################
            # Everything below is just for evaluating and saving data, 
            # not actually part of model training
            loss_values.append(loss.item())

            if i % 50 == 0:
                print(f'iteration: {i}, loss {loss.item()}')

            # Save a model from each checkpoint
            if i % 50 == 0:
                # check if the models folder exists and if not make it
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(model.state_dict(), f'./models/model_{i}.pt')
                
            i += 1
        
        clean_dataset_csv_files(in_len, target_len)

            
    
    plt.plot(loss_values)
    plt.show()

     
if __name__ == "__main__":
    main()







    


