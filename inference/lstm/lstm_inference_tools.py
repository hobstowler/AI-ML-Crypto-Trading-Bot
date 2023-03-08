print("Starting lstm_inference_tools import")
from lstm.model_tools import *
print("Finished model_tools import")
from lstm.lstm import LSTM
print("Finished LSTM import")
import torch
print("Finished lstm_inference_tools import")

# import sys

# sys.path.append('../')
# sys.path.append('../lstm')

def run_lstm_inference(hyperparams, model_path, input_tensors, pred_len):

    # Unpack hyperparameters and data source info
    input_size = hyperparams['input_size']
    hidden_size = hyperparams['hidden_size']
    output_size = hyperparams['output_size']
    num_layers = hyperparams['num_layers']
    batch_size = hyperparams['batch_size']
    # inference_source = data_info['inference_source']
    # seq_len = data_info['seq_len']
    # columns = data_info['columns']

    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, batch_size=batch_size)

    # Load model from checkpoint
    model = initialize_model(model, model_path)

    # Get data in tensor form
    #inference_tensor = tensors_from_csv(inference_source, seq_len=seq_len, columns=columns, batch_size=1)
    inference_tensor = input_tensors

    # Initialize hidden layer to zeros
    hidden_prev = model.init_hidden()

    # Save original batch size and set to 1 for prediction
    batch_size = model.batch_size
    model.set_batch_size(1)

    # Load the data from first example only
    input = inference_tensor.to(torch.float32)

    # Run through the inputs to build up the hidden state
    output, hidden_prev = model(input, hidden_prev)
    
    prediction = torch.zeros((1, pred_len, output.shape[2]))
    # Make predictions for each value in the target
    for i in range(pred_len):
        # Use the last output as the next input
        output, hidden_prev = model(output[:,-1,:].unsqueeze(1), hidden_prev)
        prediction[:,i,:] = output

    # Restore batch size to original value
    model.set_batch_size(batch_size)

    return prediction

def make_trade_decision(input, prediction, threshold=0.05, idx=0):
    """
    Generate a trade decision based on the input and prediction.
    """

    # Get the last value of the input and prediction
    print("Input: ", input)
    input_last = input[0,-1,idx]
    pred_last = prediction[0,-1,idx]

    # Calculate the percentage difference between the input and prediction
    percent_diff = (pred_last - input_last) / input_last

    # If the difference from the predicted value to current is greater than the trade threshold, buy
    if percent_diff > threshold:
        return 1
    # If the difference from the predicted value to current is less than the negative trade threshold, sell
    elif percent_diff < -threshold:
        return -1
    # Otherwise, hold
    else:
        return 0

def lstm_inference_demo(input_tensors):

    hyperparams = {
        'input_size': 2,
        'hidden_size': 80,
        'output_size': 2,
        'num_layers': 1,
        'batch_size': 1,
    }

    # data_info = {
    #     'inference_source': '../lstm/test_48.csv',
    #     'seq_len': 48,
    #     'columns': ['close_price', 'volume']
    # }

    model_path = 'lstm/models/model_test_1.pt'

    print("Starting predictions")
    predictions = run_lstm_inference(hyperparams, model_path, input_tensors, pred_len=6)
    print("Finished predictions")

    #print("Inference input", input)
    #print("Inference output", predictions)

    #trade = trade_decision(input, predictions, idx=0)

    #print("Trade decision", trade)

    return predictions
