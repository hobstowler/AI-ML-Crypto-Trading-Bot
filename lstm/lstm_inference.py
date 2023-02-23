from model_tools import *
from lstm import LSTM
import torch

def lstm_inference(hyperparams, model_path, data_info, pred_len):

    # Unpack hyperparameters
    input_size = hyperparams['input_size']
    hidden_size = hyperparams['hidden_size']
    output_size = hyperparams['output_size']
    num_layers = hyperparams['num_layers']
    batch_size = hyperparams['batch_size']

    # Unpack data source info
    inference_source = data_info['inference_source']
    seq_len = data_info['seq_len']
    columns = data_info['columns']

    model = LSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, batch_size=batch_size)

    # Load model from checkpoint
    model = initialize_model(model, model_path)

    # Get data in tensor form
    inference_tensor = tensors_from_csv(inference_source, seq_len=seq_len, columns=columns, batch_size=1)

    print("Inference input", inference_tensor[0])


    # Initialize hidden layer to zeros
    hidden_prev = model.init_hidden()

    # Save original batch size and set to 1 for prediction
    batch_size = model.batch_size
    model.set_batch_size(1)

    # Load the data
    input = inference_tensor[0].to(torch.float32)

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




if __name__ == '__main__':

    hyperparams = {
        'input_size': 2,
        'hidden_size': 80,
        'output_size': 2,
        'num_layers': 1,
        'batch_size': 1,
    }

    data_info = {
        'inference_source': './test_48.csv',
        'seq_len': 48,
        'columns': ['close_price', 'volume']
    }

    data_path = './data/test_48.csv'

    model_path = './models/model_test_1.pt'

    predictions = lstm_inference(hyperparams, model_path, data_info, pred_len=6)

    print("Inference output", predictions)
