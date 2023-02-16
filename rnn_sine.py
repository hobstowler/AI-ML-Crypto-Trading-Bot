# Adapted from: https://medium.com/dejunhuang/learning-day-27-implementing-rnn-in-pytorch-for-time-series-prediction-3ddb6190e83d

# pip install torch

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt


# For building sine data
num_time_steps = 100
context_length = 70
num_cycles = 6

# Model hyperparameters
hidden_size = 10
input_size = 1
output_size = 1
lr = 0.01

# For data collection and analysis
iter_save_values = [10, 100, 200, 300, 400, 500]
loss_values = []
pred_sets = {}


def generate_sine_data(start=None):

    radian_length = num_cycles * 2 * np.pi

    # random start point (anywhere within one full sine wave cycle)
    if start == None:
        start = np.random.randint(7, size=1)[0]
    
    time_steps = np.linspace(start, start + radian_length, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps, 1)

    return data, time_steps

def split_data(data):
    # x data is everything but last point
    x = torch.tensor(data[:-1]).float().reshape(1, num_time_steps - 1, 1)
    # y data is everything but first point
    y = torch.tensor(data[1:]).float().reshape(1, num_time_steps - 1, 1)

    return x, y

def run_test(model, x, context_length):
    
    # Initialize hidden layer to 0
    hidden_prev = torch.zeros(1, 1, hidden_size)
    preds = []

    # Let the model see the data in the context length to build hidden layer
    for idx in range(context_length):

        input_x = x[:, idx, :] # For now inputs are actual values (like training wheels)
        input_x = input_x.reshape(1,1,1)
        pred, hidden_prev = model(input_x, hidden_prev)

    # have the model predict the values following the context length
    for _ in range(context_length, x.shape[1]):
        input_x = input_x.reshape(1,1,1)

        pred, hidden_prev = model(input_x, hidden_prev)
        input_x = pred # Now the model is using its own outputs from the previous iteration as an input (no training wheels)
        preds.append(pred.detach().numpy().reshape(-1))

    return preds


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define our RNN basic architecture
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Linear layer connecting hidden layer to output ("top" of the RNN)
        self.linear = nn.Linear(hidden_size, output_size)

    # Defines what happens when the model is run. Basically the base RNN runs, then the linear layer for output runs
    def forward(self, x, hidden_prev):
        out, hidden_prev = self.rnn(x, hidden_prev)

        out = out.reshape(-1, hidden_size)

        out = self.linear(out)
        out = out.unsqueeze(dim=0)
        return out, hidden_prev

model = Net()   # build a model instance based on class we just defined
criterion = nn.MSELoss()    # Set the loss criteria as mean squared error
optimizer = optim.Adam(model.parameters(), lr)  # This will facilitate gradient updates, Adam is a very common method


# Initialize hidden layer to zeros
hidden_prev = torch.zeros(1, 1, hidden_size)

# Perform training over specified number of iterations
for iter in range(max(iter_save_values) + 1):

    # Generate new sine wave data from random start point
    data, _ = generate_sine_data()

    # Split data into x (model inputs) and y (target outputs)
    x, y = split_data(data)

    # Run model forward and get predicted values
    output, hidden_prev = model(x, hidden_prev)

    # Save the hidden layer as it will be an input for next iteration (detach removes the new copy from the gradient update graph)
    hidden_prev = hidden_prev.detach()

    # Calculate the loss from current output compared to the target values
    loss = criterion(output, y)

    # PyTorch thing, need to zero out the calculated gradients from the previous iteration, otherwise they accumulate
    model.zero_grad()

    # Run backpropogation and calculate new gradients
    loss.backward()

    # Update weights based on mainly the new gradients and learning rate
    optimizer.step()

    #########################################################################
    # Everything below is just for evaluating and saving data, not actually part of model training


    loss_values.append(loss.item())

    if iter % 50 == 0:
        print(f'iteration: {iter}, loss {loss.item()}')

    # Test a model at each given target iteration, save that models predictions for graphing later
    if iter in iter_save_values:
        data, _ = generate_sine_data(start=0)
        x, _ = split_data(data)
        pred_sets[iter] = run_test(model, x, context_length)

# Now collect predictions on a test set
data, time_steps = generate_sine_data(0)
x, _ = split_data(data)
x = x.data.numpy().reshape(-1)

# Plot ground truth
plt.subplot(211)
plt.xlabel("Radians")
plt.ylabel("Output Value")
plt.scatter(time_steps[:-1], x.reshape(-1), s=10, label='ground truth')
plt.plot(time_steps[:-1], x.reshape(-1))

# Plot output from model at each saved iteration
for iter in iter_save_values:
    if iter in pred_sets.keys():
        label = 'predicted ' + str(iter)
        plt.scatter(time_steps[context_length+1:], pred_sets[iter], label=label)
        plt.plot(time_steps[context_length+1:], pred_sets[iter])
plt.legend()

# Plot loss
plt.subplot(212)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.plot(loss_values)
plt.show()





    


