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
sys.path.append('../data')
###### Modified here for local run ######
#sys.path.append('../lstm')
print("Starting imports")
from lstm.lstm_inference_tools import lstm_inference_demo, make_trade_decision
print("Finished lstm imports in inference")
from binance_api import binance_api
print("Finished binance imports in inference")
from data_conversion.generate_datasets import inference_df_to_tensor, normalize_data, denormalize_data, manual_normalization
print("Finished data_conversion imports in inference")
from data import datastore_wrapper

SESSION_ID = 6209583034925056 

client = datastore.Client()
bp = Blueprint('inference', __name__, url_prefix='/inference')

print("Finished blueprint in inference")



def lstm_inference_pipeline(session_id=SESSION_ID):
    print("Started Inference Pipeline")
    binance = binance_api.BinanceAPI()
    print("Started binance client")
    datastore_client = datastore_wrapper.DatastoreWrapper()
    print("Started datastore client")
    columns = ['close_price', 'volume']
    # Collect recent bitcoin price data
    input_data, data_len = get_recent_binance_data(binance, context_steps=48, interval=60)

    print("Started data preprocessing")
    # Data preprocessing
    normalized_data = manual_normalization(input_data, "min_max.csv")
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
    trade_type = "Purchase" if trade_decision == 1 else "Sale"
    print("Finished decision")

    # Make trade 
    previous_price = inputs_for_prediction[-1][-2][0]
    current_price = inputs_for_prediction[-1][-1][0]
    predicted_price = denormalized_predictions[-1][-1][0]
    print("Starting trade render")
    binance.render_trade_decision("BTCUSDT", .001, trade_decision)
    binance_info = binance.get_account_balances()
    print("Finished binance account trade")

    btc_value = float(binance_info[1]['free'])
    usd_value = float(binance_info[6]['free'])

    account_value = btc_value * current_price + usd_value

    # Update datastore
    bitcoin_start = 21742
    usd_start = 10000
    # Find the step number
    step = 0
    print("Calculating step number")
    transactions= datastore_client.get_session_transactions(session_id)
    for transaction in transactions:
        if transaction['step'] > step:
            step = transaction['step']
    step += 1
    # Update datastore
    print("Starting datastore update")
    datastore_client.create_transaction(step, trade_type, 6209583034925056, **{
        "account_value": account_value-bitcoin_start-usd_start,
        "btc_actual_change": previous_price-bitcoin_start,
        "btc_predicted_change": predicted_price-bitcoin_start,
    })
    print("Finished datastore update")

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
    print("Loading data from binanace api")
    binance = binance_api.BinanceAPI()
    print("Finished loading binance api")
    data = binance.get_candlestick_dataframe(
        ticker_symbol="BTCUSDT",
        start_time=start,
        end_time=current,
        time_inteval_in_minutes=interval)
    print("Finished loading data from binance")
    data_len = len(data)
    
    print("Writing to csv")
    # Workaround for issue with data from binance that include python object types
    #data.to_csv("temp.csv", index=False)
    temp = data.to_csv()
    # Covert temp back to dataframe
    print("Temp csv string made")
    data= pd.read_csv(StringIO(temp))
    print("Finished writing to csv")

    return data, data_len

print("Running lstm inference in main")
lstm_inference_pipeline()
print("Finished running lstm inference in main")

@bp.route('/lstm', methods=['GET'])
def lstm_inference():
    if request.method == 'GET':
        print("Running LSTM Inference from endpoint")
        lstm_inference_pipeline()
        print("Finished running LSTM Inference from endpoint")
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