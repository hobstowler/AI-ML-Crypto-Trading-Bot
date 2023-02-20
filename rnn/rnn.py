# Adapted from: https://medium.com/dejunhuang/learning-day-27-implementing-rnn-in-pytorch-for-time-series-prediction-3ddb6190e83d

# IMPORTS
import torch
from torch import nn, optim
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

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
HIDDEN_SIZE = 100
INPUT_SIZE = 10
OUTPUT_SIZE = 10
NUM_LAYERS = 1
BATCH_SIZE = 1 
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

        # for training
        self.criterion = None
        self.optimizer = None
        self.loss_reduction = None

        # for data saving
        self.loss_values = []
        
        self.rnn = nn.RNN(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            batch_first = True
        )

        # Linear layer connecting hidden layer to output ("top" of the RNN)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
    
    def init_training(self, criterion, optimizer, loss_reduction):
        self.set_criterion(criterion)
        self.set_optimizer(optimizer)
        self.set_loss_reduction(loss_reduction)

    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)
        out = self.linear(out)
        # out = out.unsqueeze(dim=0)
        return out, hidden_prev
    
    def set_loss_reduction(self, value):
        self.loss_reduction = value
        
    def get_loss_reduction(self):
        return self.loss_reduction

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def get_optimizer(self):
        return self.optimizer

    def set_criterion(self, criterion):
        self.criterion = criterion
        
    def get_criterion(self):
        return self.criterion
    
    def set_batch_size(self, value: int):
        self.batch_size = value

    def generate_dataset(self, sequence_length):
        generate_csv_datasets("../training_data/2021.csv", 
                              seq_len=sequence_length)
        
    def clean_dataset_csvs(self, sequence_length):
        clean_dataset_csv_files(sequence_length)

    def get_tensors(self, sequence_length, batch_sz, kind: str="train"):
        tensors = None
        try:
            tensors = tensors_from_csv(
            f"{kind}_{sequence_length}.csv", 
            seq_len=48, 
            columns=[
                "open_price", "high_price", "low_price", "close_price", 
                "volume", "close_time", "quote_asset_volume","qty_transactions", 
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"
            ], 
            batch_size=batch_sz
            )
        except:
            pass
        return tensors

    def train(self, epochs, lr, optimizer, criterion, sequence_length=48):
        
        self.init_training(optimizer=optimizer, criterion=criterion, 
                           loss_reduction=lr)
        i = 0
        for epoch in range(epochs):
            
            self.generate_dataset(sequence_length=sequence_length)
            train_tensors = self.get_tensors(sequence_length, 
                                            batch_sz=self.batch_size, 
                                            kind="train")

            # Initialize hidden layer to zeros
            hidden_prev = self.init_hidden()

            # Perform training over specified number of iterations
            for tensor in train_tensors:
            
                tensor = tensor.to(torch.float32)
                inputs, targets = tensor[:,:-1,:], tensor[:,1:,:]
                output, hidden_prev = self(inputs, hidden_prev)
                hidden_prev = hidden_prev.detach()
                loss = self.get_criterion()(output, targets)
                self.zero_grad()
                loss.backward()
                self.get_optimizer().step()
                self.loss_values.append(loss.item())
                if i % 1 == 0:
                    print(f'iteration: {i}, loss {loss.item()}')

                # Save a model from each checkpoint
                # if i % 50 == 0:
                #     # check if the models folder exists and if not make it
                #     if not os.path.exists('./models'):
                #         os.makedirs('./models')
                #     torch.save(model.state_dict(), f'./models/model_{i}.pt')
                
                i += 1
        
        
        plt.plot(self.loss_values)
        plt.show()
    
    def validate(self, sequence_length, batch_size):
        # Initialize hidden layer to zeros
        hidden_prev = self.init_hidden()

        data_tensors = self.get_tensors(sequence_length=sequence_length, 
                                        batch_sz=batch_size, kind="val")
        
        preds = []
        preds_targets = []

        iter = 0
        with torch.no_grad():
            for curr_tensor in data_tensors:

                curr_tensor = curr_tensor.to(torch.float32)

                # Split into input and target values
                inputs, targets = curr_tensor[:,:-1,:], curr_tensor[:,1:,:]
                preds_targets.append(targets)
                output, hidden_prev = self(inputs, hidden_prev)
                hidden_prev = hidden_prev.detach()
                preds.append(output)
                loss = self.get_criterion()(output, targets)

                self.loss_values.append(loss.item())

                if iter % 1 == 0:
                    print(f'iteration: {iter}, validation loss {loss.item()}')

                iter += 1

        pred_x = []
        pred_y = []
        
        for tensor in preds:
            tensor_list = tensor.squeeze().tolist()
            for pred in tensor_list:
                pred_x.append(pred[0])
        for tensor in preds_targets:
            tensor_list = tensor.squeeze().tolist()
            for pred in tensor_list:
                pred_y.append(pred[0])
        plt.plot(pred_x, 'b')
        plt.plot(pred_y, 'y')
        plt.show()

        
    
    def predict(self, sequence_length, batch_size, pred_len):
        # Initialize hidden layer to zeros

        self.set_batch_size(1)

        data_tensors = self.get_tensors(sequence_length=sequence_length, 
                                        batch_sz=batch_size, kind="test")

        iter = 0
        predictions = []
        with torch.no_grad():
            for curr_tensor in data_tensors:
                hidden_prev = self.init_hidden()

                curr_tensor = curr_tensor.to(torch.float32)

                split_point = len(curr_tensor[0]) - pred_len

                # Split into input and target values
                inputs = curr_tensor[:, :split_point, :] 
                targets = curr_tensor[:, split_point:, :]
                # Run through the inputs to build up the hidden state
                output, hidden_prev = self(inputs, hidden_prev)
            
                prediction = torch.zeros(targets.shape)
                # Make predictions for each value in the target
                for i in range(pred_len):
                    # Use the last output as the next input
                    output, hidden_prev = self(output[:,-1,:].unsqueeze(1), 
                                               hidden_prev)
                    prediction[:,i,:] = output

                loss = self.get_criterion()(prediction, targets)

                self.loss_values.append(loss.item())

                if iter % 1 == 0:
                    print(f'iteration: {iter}, prediction loss {loss.item()}')

                iter += 1

                print("Prediction: ", prediction)
                print("Target: ", targets)

                predictions.append(prediction)

        return predictions

    def vizualize_predictions(self, predictions, targets, index=0):

        grid_rows = int(np.ceil(np.sqrt(len(predictions))))
        grid_cols = int(np.ceil(len(predictions)/grid_rows))

        figure, axis = plt.subplots(grid_rows, grid_cols)

        for row in range(grid_rows):
            for col in range(grid_cols):
                index = (row * grid_cols) + col
                if index < len(predictions):
                    x_targets = np.arange(0, len(targets[0].squeeze().numpy()))
                    start_point = len(targets[0].squeeze().numpy())\
                                - len(predictions[index].squeeze().numpy())
                    x_predictions = np.arange(start_point, len(x_targets))
                    
                    # print("x_targets\n", x_targets)
                    # print("targets[index]\n", targets[index].squeeze().numpy())
                    # return
                    axis[row][col].scatter(
                        x_targets, targets[index].squeeze().numpy())
                    axis[row][col].scatter(
                        x_predictions, predictions[index].squeeze().numpy())
                

        plt.show()


def main():

    # build rnn model
    model = Net(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        output_size=OUTPUT_SIZE, 
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE
    )

    # train model
    model.train(
        epochs=EPOCHS, 
        lr=LR,
        optimizer=optim.Adam(model.parameters(), LR),
        criterion=nn.MSELoss(),
        sequence_length=48,
    )

    # validate model
    model.validate(sequence_length=48, batch_size=BATCH_SIZE)
    
    # predictions
    # preds = model.predict(
    #     sequence_length=48, batch_size=BATCH_SIZE, pred_len=12)
    # targets = model.get_tensors(
    #     sequence_length=48, batch_sz=BATCH_SIZE, kind="test")
    # model.vizualize_predictions(
    #     predictions=preds, targets=targets, index=7)

    model.clean_dataset_csvs(48)

    
    
    
if __name__ == "__main__":
    main()
