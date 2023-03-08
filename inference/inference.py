from flask import Blueprint, request, jsonify
from google.cloud import datastore
import os
import sys
import datetime
import pandas as pd
from io import StringIO

# Add parent directory to path so we can import from data_conversion
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)

sys.path.append('../')
sys.path.append('../binance_api')
sys.path.append('../data_conversion')
###### Modified here for local run ######
#sys.path.append('../lstm')
print("Starting imports")
from lstm.lstm_inference_tools import lstm_inference_demo, make_trade_decision
print("Finished lstm imports in inference")
from binance_api import binance_api
print("Finished binance imports in inference")
from data_conversion.generate_datasets import inference_df_to_tensor, normalize_data, denormalize_data, manual_normalization
print("Finished data_conversion imports in inference")


client = datastore.Client()
bp = Blueprint('inference', __name__, url_prefix='/inference')

print("Finished blueprint in inference")



def lstm_inference_pipeline():
    print("Started Inference Pipeline")
    columns = ['close_price', 'volume']
    # Collect recent bitcoin price data
    input_data, data_len = get_recent_binance_data(context_steps=48, interval=60)

    print("Started data preprocessing")
    # Data preprocessing
    normalized_data = manual_normalization(input_data)
    print("Normalized data", normalized_data)
    input_tensors = inference_df_to_tensor(normalized_data, data_len, columns)

    print("Started inference")
    # Run inference
    predictions = lstm_inference_demo(input_tensors)

    print("Started post processing")
    # Post processing (denormalization)
    denormalized_predictions = denormalize_data(predictions, 'min_max.csv', columns)
    inputs_for_prediction = denormalize_data(input_tensors, 'min_max.csv', columns)

    print("Started trade decision")
    # Make trade decision
    trade_decision = make_trade_decision(inputs_for_prediction, denormalized_predictions)

    print("Inputs for prediction", inputs_for_prediction)
    print("Predictions", denormalized_predictions)
    print("Trade decision", trade_decision)

    # Update Binance account

    # Update datastore

    return '', 200

def get_recent_binance_data(context_steps=48, interval=60):
    # get most recent times
    current = datetime.datetime.utcnow()

    start = current - datetime.timedelta(minutes=context_steps*interval)
    print("Time delta", datetime.timedelta(minutes=context_steps*interval))
    print(current)
    print(start)
    start = datetime.datetime.strftime(
        start, "%Y-%m-%d %H:%M:%S")
    current = datetime.datetime.strftime(
        current, "%Y-%m-%d %H:%M:%S")

    # get recent data from binance
    binance = binance_api.BinanceAPI()
    data = binance.get_candlestick_dataframe(
        ticker_symbol="BTCUSDT",
        start_time=start,
        end_time=current,
        time_inteval_in_minutes=interval)
    data_len = len(data)
    
    print("Writing to csv")
    # Workaround for issue with data from binance that include python object types
    #data.to_csv("temp.csv", index=False)
    temp = data.to_csv()
    # Covert temp back to dataframe
    print("Temp csv string made")
    #data = pd.read_csv(temp)
    data= pd.read_csv(StringIO(temp))
    #os.remove("temp.csv")
    print("Finished writing to csv")

    return data, data_len

print("Running inference pipeline")
lstm_inference_pipeline()
print("Finished inference pipeline")

@bp.route('/lstm', methods=['GET'])
def lstm_inference():
    if request.method == 'GET':
        print("Running LSTM Inference")
        lstm_inference_pipeline()
        return '', 200

@bp.route('/rnn', methods=['GET'])
def rnn_inference():
    if request.method == 'GET':
        # Add rnn inference here

        return '', 200

@bp.route('/rl', methods=['GET'])
def rl_inference():
    if request.method == 'GET':
        # Add rl inference here

        return '', 200