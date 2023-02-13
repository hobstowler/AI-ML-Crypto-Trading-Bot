# Adapted from: https://medium.com/dejunhuang/learning-day-27-implementing-rnn-in-pytorch-for-time-series-prediction-3ddb6190e83d

# pip install torch

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt


# For building sine data
NUM_TIME_STEPS = 80
CONTEXT_LENGTH = 70
NUM_CYCLES = 6

# Model hyperparameters
HIDDEN_SIZE = 10
INPUT_SIZE = 1
OUTPUT_SIZE = 1
LR = 0.001

# For data collection and analysis
MODEL_SAVE_POINTS = [10, 100, 200, 500, 1000]
loss_values = []
pred_sets = {}


# Make data for model to train or predict with
def generate_sine_data(start=None, num_time_steps=100, num_cycles=6):

    radian_length = num_cycles * 2 * np.pi

    # random start point (anywhere within one full sine wave cycle)
    if start == None:
        start = np.random.randint(7, size=1)[0]
    
    time_steps = np.linspace(start, start + radian_length, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)

    return data, time_steps

# Splits data into context info and target values
def split_format_data(data, CONTEXT_LENGTH=80):

    x = torch.tensor(data[:CONTEXT_LENGTH]).float().reshape(1, CONTEXT_LENGTH, 1)
    y = torch.tensor(data[CONTEXT_LENGTH:]).float().reshape(1, len(data) - CONTEXT_LENGTH, 1)

    return x, y

# Encoder model that reads historical data and encodes it into a hidden vector
class rnn_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(rnn_encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout = 0     # Dropout is only applied between stacked RNN layers, need to have more than 1
        )

    # Initialize a hidden state of zeros for the encoder to start with
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.num_layers, self.hidden_size)

    def forward(self, x, hidden_prev):
        hidden_prev = self.rnn(x, hidden_prev)

        return hidden_prev

# Function that handles the training of the model
def train_model(model, iterations, LR, model_save_points):
    for iter in range(iterations):
        data, _ = generate_sine_data()
        x, y = split_format_data(data)
        target_length = len(y[0])
        model.train(x, y, target_length, LR, teacher_forcing=False, loss_values=loss_values)

        # For every point in the MODEL_SAVE_POINTS list, save the model
        if iter in model_save_points:
            torch.save(model.state_dict(), "./saved_models/model_{}.pt".format(iter))


# Decoder model that takes the hidden vector from the encoder and predicts future values
class rnn_decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(rnn_decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(
            input_size=self.output_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            dropout = 0     # Dropout is only applied between stacked RNN layers, need to have more than 1
        )

        self.linear = nn.Linear(self.hidden_size, self.output_size)

    # Forward pass accepts the hidden state from the encoder
    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)

        out = out.reshape(-1, self.hidden_size)

        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev

# Combining the encoder and decoder into one model
class encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(encoder_decoder, self).__init__()

        self.encoder = rnn_encoder(input_size, hidden_size)
        self.decoder = rnn_decoder(hidden_size, output_size)

    def train(self, inputs, targets, target_len, LR, teacher_forcing=False, loss_values=[]):

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), LR)

        # Make a tensor for storing the outputs
        outputs = torch.zeros(targets.shape)

        encoder_hidden = self.encoder.init_hidden(1)

        # Run encoder forward to get hidden layer
        encoder_hidden = self.encoder(inputs, encoder_hidden)

        # Handoff the hidden vector
        decoder_hidden = encoder_hidden[1]

        # Make the first input for the decoder the last item from input values
        decoder_input = inputs[:, -1, :]
        decoder_input = decoder_input.reshape(1,1,1)
        
        if not teacher_forcing:
            # Run the decoder forward recursively without teacher forcing
            for ind in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # Make the next input the last output
                outputs[0,ind,0] = decoder_output # Make the first selection batch number later
                decoder_input = decoder_output

        # Case for teacher forcing
        else:
            # Run the decoder forward recursively with teacher forcing
            for ind in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

                # Get the next input from true data
                outputs[0,ind,0] = decoder_output
                decoder_input = targets[0, ind, 0]
                decoder_input = decoder_input.reshape(1,1,1)

        # Calculate loss
        loss = criterion(outputs, targets)
        loss_values.append(loss.item())

        print(loss.item())

        self.zero_grad()

        loss.backward()
        optimizer.step()

    def predict(self, inputs, output_length):
        # Make a tensor for storing the outputs
        outputs = torch.zeros(1, output_length, 1)

        encoder_hidden = self.encoder.init_hidden(1)

        # Run encoder forward to get hidden layer
        encoder_hidden = self.encoder(inputs, encoder_hidden)

        # Handoff the hidden vector
        decoder_hidden = encoder_hidden[1]

        # Make the first input for the decoder the last item from input values
        decoder_input = inputs[:, -1, :]
        decoder_input = decoder_input.reshape(1,1,1)

        # Run the decoder forward recursively without teacher forcing
        for ind in range(output_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # Make the next input the last output
            outputs[0,ind,0] = decoder_output

        return outputs


model = encoder_decoder(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)   # Initialize the model

iterations = max(MODEL_SAVE_POINTS) + 1

train_model(model, iterations, 0.001, MODEL_SAVE_POINTS)

# Now collect predictions on a test set
data, time_steps = generate_sine_data(start=0, num_time_steps=NUM_TIME_STEPS, num_cycles=NUM_CYCLES)
x, _ = split_format_data(data, CONTEXT_LENGTH)


# Plot the ground truth 
plt.subplot(211)
plt.xlabel("Radians")
plt.ylabel("Output Value")
plt.scatter(time_steps[:CONTEXT_LENGTH], x.reshape(-1), s=10, label='ground truth')
plt.plot(time_steps[:CONTEXT_LENGTH], x.reshape(-1))

# For every saved model point, load the model and make predictions
for iter in MODEL_SAVE_POINTS:
    model.load_state_dict(torch.load("./saved_models/model_{}.pt".format(iter)))

    outputs = model.predict(x, NUM_TIME_STEPS - CONTEXT_LENGTH)

    # Plot the model output
    outputs = outputs.detach().numpy().reshape(-1)
    label = 'predicted ' + str(iter)
    plt.scatter(time_steps[CONTEXT_LENGTH:], outputs, s=10, label=label)
    plt.plot(time_steps[CONTEXT_LENGTH:], outputs)

plt.legend()

# Plot loss
plt.subplot(212)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.plot(loss_values)
plt.show()





    


