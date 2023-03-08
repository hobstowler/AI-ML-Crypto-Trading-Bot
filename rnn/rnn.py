import torch
from torch import nn
import os
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from data_conversion.generate_datasets import tensors_from_csv
from data_conversion.generate_datasets import generate_csv_datasets
from data_conversion.generate_datasets import clean_dataset_csv_files
from data_conversion.generate_datasets import normalize_data
from binance_api import binance_api
from sklearn.preprocessing import MinMaxScaler


# Model hyperparameters
HIDDEN_SIZE = 100
INPUT_SIZE = 10
OUTPUT_SIZE = 10
NUM_LAYERS = 5
BATCH_SIZE = 1 
LR = 1e-5
EPOCHS = 50


class Net(nn.Module):

    def __init__(
        self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, 
        output_size=OUTPUT_SIZE, num_layers=NUM_LAYERS, batch_size=BATCH_SIZE
    ):
        super(Net, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()

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
        generate_csv_datasets("training_data/2021.csv", 
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

    def _train(self, epochs, lr, optimizer, criterion, sequence_length=48):
        
        self.init_training(optimizer=optimizer, criterion=criterion, 
                           loss_reduction=lr)
        i = 0
        for epoch in range(epochs):
            
            self.scaler = self.generate_dataset(sequence_length=sequence_length)
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

    def save(self):
        current_time = str(datetime.datetime.now())
        torch.save(self.state_dict(), f"rnn/saved_rnn_models/{current_time}")

    def load(self, path_to_rnn_model):
        self.load_state_dict(torch.load(path_to_rnn_model))
        self.eval()
        
    def normalize_columns(self, dataframe):
        dataframe, self.scaler = normalize_data(dataframe, self.scaler)
        return dataframe
    
    def denormalize_close_price(self, close_price_norm):
        min_val = self.scaler.data_min_[3]
        max_val = self.scaler.data_max_[3]
        diff = max_val - min_val
        close_price_denorm = (close_price_norm * diff) + min_val
        return close_price_denorm

    def get_next_close_price(self):
        
        # get most recent times
        today = datetime.datetime.now()
        yesterday = today - datetime.timedelta(days=1)
        yesterday = datetime.datetime.strftime(
            yesterday, "%Y-%m-%d %H:%M:%S")
        today = datetime.datetime.strftime(
            today, "%Y-%m-%d %H:%M:%S")
        
        # get data from binance
        binance = binance_api.BinanceAPI()
        data = binance.get_candlestick_dataframe(
            ticker_symbol="BTCUSDT",
            start_time=yesterday,
            end_time=today,
            time_inteval_in_minutes=60)
        data_len = len(data)

        # convert data to tensors
        current_time = str(
            datetime.datetime.now().strftime("%Y%m%d %H:%M:%S%f"))
        data_filepath = f"./data-{current_time}"
        binance.export_candlestick_dataframe_csv(
            pandas_dataframe=data,
            csv_filepath=data_filepath)
        df = pd.read_csv(data_filepath)
        # print("DF BEFORE NORM\n", df)
        df = self.normalize_columns(df)
        # print("DF\n", df)
        df.to_csv(data_filepath, index=False)
        tensors = tensors_from_csv(
            infile=data_filepath, 
            seq_len=data_len,
            columns=[
                "open_price", "high_price", "low_price", 
                "close_price", "volume", "close_time", 
                "quote_asset_volume","qty_transactions", 
                "taker_buy_base_asset_volume", 
                "taker_buy_quote_asset_volume"
            ])
        # print("TENSORS\n", tensors)

        # delete data csv
        os.remove(data_filepath)
        
        # run model
        hidden_prev = self.init_hidden()
        for tensor in tensors:
            tensor = tensor.to(torch.float32)
            output, hidden_prev = self(tensor, hidden_prev)
            hidden_prev = hidden_prev.detach()
            
        # convert output back to datafram for denormalization
        output = output.detach().squeeze()
        output =  pd.DataFrame(output.numpy())
        # print("OUTPUT:\n", output)
        # print("DENORM open price last row\n", output.iloc[-1, 0])
        next_close_price = self.denormalize_close_price(output.iloc[-1, 3])
        
        return next_close_price

    def get_trade_decision(self):
        binance = binance_api.BinanceAPI()
        btc_info = binance.get_client().get_symbol_ticker(symbol="BTCUSDT")
        current_price = float(btc_info.get("price"))
        print(f"current price: {current_price}")
        next_close = self.get_next_close_price()
        print(f"next close: {next_close}")
        diff = next_close - current_price
        pct_change = (diff / current_price) * 100
        print(f"percent change: {pct_change}")
        if pct_change > 0.5:
            return 1
        if pct_change < -0.5:
            return -1
        return 0
    
    def load_alpha(self):
        self.load("rnn/saved_rnn_models/alpha.rnn")
