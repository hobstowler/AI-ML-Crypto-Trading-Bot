from flask import Blueprint, request, jsonify
from google.cloud import datastore
import os
import sys
import datetime
import pandas as pd
from io import StringIO
import time

sys.path.append('../')
sys.path.append('../binance_api')
sys.path.append('../data_conversion')
sys.path.append('../data')
from lstm.lstm_inference_tools import lstm_inference_demo, make_trade_decision
from binance_api import binance_api
from data_conversion.generate_datasets import inference_df_to_tensor, normalize_data, denormalize_data, manual_normalization
from data import datastore_wrapper

SESSION_ID = 4920093460922368
BITCOIN_START = 22087
USD_START = 8322

client = datastore.Client()
bp = Blueprint('inference', __name__, url_prefix='/inference')

def lstm_inference_pipeline(session_id=SESSION_ID):
    binance = binance_api.BinanceAPI()
    datastore_client = datastore_wrapper.DatastoreWrapper()
    columns = ['close_price', 'volume']

    # Collect recent bitcoin price data
    input_data, data_len = get_recent_binance_data(binance, context_steps=48, interval=60)

    # Data preprocessing
    normalized_data = manual_normalization(input_data, "min_max.csv")
    input_tensors = inference_df_to_tensor(normalized_data, data_len, columns)

    # Run inference
    predictions = lstm_inference_demo(input_tensors)

    # Post processing (denormalization)
    denormalized_predictions = denormalize_data(predictions, 'min_max.csv', columns)
    inputs_for_prediction = denormalize_data(input_tensors, 'min_max.csv', columns)

    # Make trade decision
    trade_decision = make_trade_decision(inputs_for_prediction, denormalized_predictions)
    trade_type = "Purchase" if trade_decision == 1 else "Sale"
    trade_direction = 100 if trade_decision == 1 else -100

    # Make trade 
    previous_price = inputs_for_prediction[-1][-2][0]
    current_price = inputs_for_prediction[-1][-1][0]
    predicted_price = denormalized_predictions[-1][-1][0]
    binance.render_trade_decision("BTCUSDT", .001, trade_decision)
    binance_info = binance.get_account_balances()

    # Calculate values for use in datastore
    btc_value = float(binance_info[1]['free'])
    usd_value = float(binance_info[6]['free'])
    net_value = btc_value * current_price + usd_value

    # Find the step number for use in datastore
    step = 0
    transactions= datastore_client.get_session_transactions(session_id)
    for transaction in transactions:
        if transaction['step'] > step:
            step = transaction['step']
    step += 1

    #Update datastore
    datastore_client.create_transaction(step, trade_type, session_id, **{
        "trade_direction": trade_direction,
        "account_value": net_value-BITCOIN_START-USD_START,
        "btc_last_change": current_price-previous_price,
        "btc_predicted_change": predicted_price-current_price,
    })

    return '', 200

def get_recent_binance_data(binance, context_steps=48, interval=60):
    # get most recent times
    current = datetime.datetime.utcnow()

    start = current - datetime.timedelta(minutes=context_steps*interval)
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
    temp = data.to_csv()
    # Covert temp back to dataframe
    data= pd.read_csv(StringIO(temp))

    return data, data_len

@bp.route('/lstm', methods=['GET'])
def lstm_inference():
    if request.method == 'GET':
        lstm_inference_pipeline()
        return '', 200
    

if __name__ == '__main__':
    # Loop to run inference locally
    REFRESH_MINUTES = 60
    while True:
        lstm_inference_pipeline()
        time.sleep(REFRESH_MINUTES * 60)
