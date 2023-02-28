from binance_api.binance_api import BinanceAPI

        
def demo_binance_api():
    binance = BinanceAPI()
    
    # BALANCES
    account_info = binance.get_account_information()
    print("\n---- Account Info ----")
    print(account_info)
    balances = binance.get_account_balances()
    print("\n---- ALL BALANCES ----")
    print(balances)
    btc_balance = binance.get_account_balances(["BTC"])
    print("\n---- BTC BALANCE ----")
    print(btc_balance)
    usdt_balance = binance.get_account_balances(["USDT"])
    print("\n---- USDT BALANCE ----")
    print(usdt_balance)
    
    # BUY
    order = binance.buy_asset("BTCUSDT", 0.001)
    print("\n---- ORDER RESULTS: BUY 0.001 BTC/USDT ----")
    print(order)
    btc_balance = binance.get_account_balances(["BTC"])
    print("\n---- BTC BALANCE ----")
    print(btc_balance)
    usdt_balance = binance.get_account_balances(["USDT"])
    print("\n---- USDT BALANCE ----")
    print(usdt_balance)
    
    # SELL
    order = binance.sell_asset("BTCUSDT", 0.001)
    print("\n---- ORDER RESULTS: SELL 0.001 BTC/USDT ----")
    print(order)
    btc_balance = binance.get_account_balances(["BTC"])
    print("\n---- BTC BALANCE ----")
    print(btc_balance)
    usdt_balance = binance.get_account_balances(["USDT"])
    print("\n---- USDT BALANCE ----")
    print(usdt_balance)
