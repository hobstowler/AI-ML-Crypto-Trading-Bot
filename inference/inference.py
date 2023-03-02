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
from lstm.lstm_inference_tools import lstm_inference_demo
from binance_api import binance_api
from data_conversion.generate_datasets import inference_df_to_tensor, normalize_data


client = datastore.Client()
bp = Blueprint('inference', __name__, url_prefix='/inference')

@bp.route('/lstm', methods=['GET'])
def lstm_inference():
    if request.method == 'GET':
        return lstm_inference_pipeline()


def lstm_inference_pipeline():
    # Collect recent bitcoin price data
    data, data_len = get_recent_binance_data()

    # Data preprocessing
    normalized_data = normalize_data(data)
    input_tensors = inference_df_to_tensor(normalized_data, data_len, ['close_price', 'volume'])

    # Run inference
    lstm_inference_demo(input_tensors)

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
    df.to_csv("temp.csv", index=False)
    df = pd.read_csv("temp.csv")
    os.remove("temp.csv")

    return data, data_len
