import pandas as pd
import requests

from project import Project

project = Project()

#binance_secret = project.get_secret("binance-api-secret-hobs")
#binance_key = project.get_secret("binance-api-key-hobs")

symbol = 'BTCUSDT'
interval = '1m'
resp = requests.get(f'https://api.binance.us/api/v3/klines?symbol={symbol}&interval={interval}')
if resp.status_code == 200:
    print('OK')
    json = resp.json()
    for candle in json:
        print(candle)