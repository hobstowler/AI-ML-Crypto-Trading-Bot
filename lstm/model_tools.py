# Description: Tools to support model training and testing for supervised models.

# pip install torch
# pip install matplotlib
# pip install numpy

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent path to directory so we can import from data_conversion
sys.path.append('../')

from data_conversion.generate_datasets import tensors_from_csv

def save_model(model, optimizer, save_path):
    """
    Save the model and optimizer state dictionaries to a file.
    """

    # check if the models folder exists and if not make it
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

def predict(model, criterion, data_tensors, pred_len):
    """
    Run predictions on a test set
    """

    # Initialize hidden layer to zeros

    loss_values= []
    total_loss = 0

    # Save original batch size and set to 1 for prediction
    batch_size = model.batch_size
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
            total_loss += loss.item()

            iter += 1

            predictions.append(prediction)

    # Restore batch size to original value
    model.set_batch_size(batch_size)

    return predictions, (total_loss / iter)

def vizualize_predictions(predictions, targets, feature_idx=0):
    """
    Generate plots from the predictions and targets for display.
    """

    grid_rows = int(np.ceil(np.sqrt(len(predictions))))
    grid_cols = int(np.ceil(len(predictions)/grid_rows))
    

    figure, axis = plt.subplots(grid_rows, grid_cols)

    figure.suptitle('Predictions vs Targets', fontsize=16)

    for row in range(grid_rows):
        for col in range(grid_cols):
            index = row*grid_cols + col
            if index < len(predictions):
                # Build the x-values for targets
                x_targets = np.arange(0,len(targets[0].squeeze().numpy()))

                # Build the x-values for predictions
                start_point = len(targets[0].squeeze().numpy()) - len(predictions[index].squeeze().numpy())
                x_predictions = np.arange(start_point, len(x_targets))

                axis[row][col].scatter(x_targets, targets[index].squeeze().numpy()[:,feature_idx])
                axis[row][col].scatter(x_predictions, predictions[index].squeeze().numpy()[:,feature_idx])
                
    plt.savefig('predictions.png')

    plt.show()

def initialize_model(model, load_path):
    """
    Initialize a model from a saved checkpoint
    """
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model

def train_lstm(model, criterion, data_tensors, optimizer, save_points):
    """
    Perform training over one epoch of data.
    """

    # Initialize hidden layer to zeros
    hidden_prev = model.init_hidden()

    total_loss = 0 
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

        total_loss += loss.item()

        # Save a model from each checkpoint
        if iter in save_points:
            save_model(model, optimizer, f'./models/model_{iter}.pt')

        iter += 1

    return total_loss / iter

def validate_lstm(model, criterion, data_tensors):
    """
    Perform validation over one epoch of data.
    """

    # Initialize hidden layer to zeros
    hidden_prev = model.init_hidden()

    total_loss = 0
    iter = 0
    with torch.no_grad():
        for curr_tensor in data_tensors:

            curr_tensor = curr_tensor.to(torch.float32)

            # Split into input and target values
            inputs, targets = curr_tensor[:,:-1,:], curr_tensor[:,1:,:]
            output, hidden_prev = model(inputs, hidden_prev)
            hidden_prev = (hidden_prev[0].detach(), hidden_prev[1].detach())
            loss = criterion(output, targets)

            total_loss += loss.item()

            iter += 1
    
    return total_loss / iter