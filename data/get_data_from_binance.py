import os
import sys
from datetime import datetime, timezone
import pytz

import pandas as pd
import requests

from project import Project

#project = Project()

#binance_secret = project.get_secret("binance-api-secret-hobs")
#binance_key = project.get_secret("binance-api-key-hobs")

""" ###############################################################
    PARAMETERS
############################################################### """

# TODO clean up and make programmatic
train_start_dt_tm = '2017-01-01 00:00:00'
train_start_dt = int(datetime.strptime(train_start_dt_tm, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
train_end_dt_tm = '2022-12-31 23:59:00'
train_end_dt = int(datetime.strptime(train_end_dt_tm, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
test_start_dt_tm = '2022-01-01 00:00:00'
test_start_dt = int(datetime.strptime(test_start_dt_tm, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)
test_end_dt_tm = '2022-12-31 23:59:00'
test_end_dt = int(datetime.strptime(test_end_dt_tm, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp() * 1000)

train = True
test = True

symbol = 'BTCUSD'
interval = '1m'

if interval == '1m':
    interval_time = 1000 * 60
elif interval == '5m':
    interval_time = 1000 * 60 * 5
elif interval == '30m':
    interval_time = 1000 * 60 * 30
elif interval == '1h':
    interval_time = 1000 * 60 * 60
elif interval == '1d':
    interval_time = 1000 * 60 * 60 * 24
else:
    print("bad interval")
    sys.exit()

max_candles = 1000
columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'asset volume', 'num trades', 'base asset volume', 'quote asset volume', 'meh']
#df = pd.DataFrame(columns=columns)

if train:
    dataframes = []
    for i in range(train_start_dt, train_end_dt, interval_time * max_candles):
        start = i
        end = i + (interval_time * max_candles) - interval_time
        end = end if end <= train_end_dt else train_end_dt
        resp = requests.get(f'https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={max_candles}&startTime={start}&endTime={end}')
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json(), columns=columns)
            dataframes.append(df)
            #df = df.append(ndf, ignore_index=False)
        elif resp.status_code == 429:
            print(f'error 429: rate limit tripped at i={i}. start: {start}, end: {end}')
            print(resp.json())
            break

    df = pd.concat(dataframes, ignore_index=True)
    df.to_csv(f'{os.getcwd()}/train_{train_start_dt_tm[:10]}_{train_end_dt_tm[:10]}_{interval}.csv', index=False)

if test:
    dataframes = []
    for i in range(test_start_dt, test_end_dt, interval_time * max_candles):
        start = i
        end = i + (interval_time * max_candles) - interval_time
        end = end if end <= test_end_dt else test_end_dt
        resp = requests.get(f'https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}&limit={max_candles}&startTime={start}&endTime={end}')
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json(), columns=columns)
            dataframes.append(df)
            #df = df.append(ndf, ignore_index=False)
        elif resp.status_code == 429:
            print(f'error 429: rate limit tripped at i={i}. start: {start}, end: {end}')
            print(resp.json())
            break

    df = pd.concat(dataframes, ignore_index=True)
    df.to_csv(f'{os.getcwd()}/test_{test_start_dt_tm[:10]}_{test_end_dt_tm[:10]}_{interval}.csv', index=False)
