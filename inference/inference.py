from flask import Blueprint, request, jsonify
from google.cloud import datastore
import os
import sys
import datetime
import pandas as pd

# Add parent directory to path so we can import from data_conversion
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(current)
# sys.path.append(parent)

sys.path.append('../')
sys.path.append('../binance_api')
sys.path.append('../data_conversion')
from lstm.lstm_inference_tools import lstm_inference_demo, make_trade_decision
from binance_api import binance_api
from data_conversion.generate_datasets import inference_df_to_tensor, normalize_data


client = datastore.Client()
bp = Blueprint('inference', __name__, url_prefix='/inference')

@bp.route('/lstm', methods=['GET'])
def lstm_inference():
    if request.method == 'GET':
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

def lstm_inference_pipeline():
    # Collect recent bitcoin price data
    input_data, data_len = get_recent_binance_data(context_steps=48, interval=60)

    # Data preprocessing
    normalized_data = normalize_data(input_data)
    input_tensors = inference_df_to_tensor(normalized_data, data_len, ['close_price', 'volume'])

    # Run inference
    predictions = lstm_inference_demo(input_tensors)

    # Post processing (denormalization)

    # Make trade decision
    trade_decision = make_trade_decision(input_tensors, predictions)

    print("Input data", input_tensors)
    print("Predictions", predictions)
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
    
    # Workaround for issue with data from binance that include python object types
    data.to_csv("temp.csv", index=False)
    data = pd.read_csv("temp.csv")
    os.remove("temp.csv")

    return data, data_len
