from scripts.rnn_next_candle import rnn_next_candle
from scripts.demo_binance_api import demo_binance_api
from scripts.export_candlestick_data import export_candlestick_data
from scripts.train_rnn import train_rnn

def main():
#    export_candlestick_data(
#        ticker_symbol="BTCUSDT", 
#        start_time="2020-01-01 00:00:00",
#        end_time="2020-12-31 23:59:59",
#        time_interval_in_minutes=60,
#        csv_filepath="training_data/2020.csv"
#    ) 
   demo_binance_api()
#    train_rnn()
   rnn_next_candle()

if __name__ == "__main__":
    main()