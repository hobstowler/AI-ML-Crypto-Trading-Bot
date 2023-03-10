from rnn.rnn import Net
from data.datastore_wrapper import DatastoreWrapper
from binance_api.binance_api import BinanceAPI
from datetime import datetime

SESSION_ID = None
SESSION_NAME = "RNN Portfolio"
TYPE = "Trading Simulation"
MODEL_NAME = "RNN Model"
CRYPTO_TYPE = "BTC/USDT"
# SESSION_START = 1
# SESSION_END = None
# STARTING_BALANCE = 10
# ENDING_BALANCE = 3
# STARTING_COINS = 20
# ENDING_COINS = 30
# COINS_BOUGHT = 5
# COINS_SOLD = 4
# NUMBER_OF_TRADES = 6


rnn = Net()
datastore = DatastoreWrapper()
binance = BinanceAPI()

def rnn_portfolio(step_num: int):
    print()
    id = SESSION_ID
    session_exists = check_session_exists()
    if not session_exists:
        print("Session does not exist.\nCreating new session.")
        id = create_session()
    print("Loading RNN Alpha Model.")
    rnn.load_alpha()
    print("Requesting trading decision from RNN...")
    rnn_trade_decision = rnn.get_trade_decision()
    print(f"Trade decision -> {rnn_trade_decision}")
    binance.render_trade_decision(
        symbol="BTCUSDT",
        quantity=0.001,
        trade_decision=rnn_trade_decision
    )
    update_datastore_session(id, rnn_trade_decision, step_num)
    
def reset_balances():
    btc_balance = float(get_balance_btc())
    btc_to_sell = round((btc_balance - 1), 8)
    # print(f"bitcoin to sell: {btc_to_sell}")
    if btc_to_sell > 0:
        trade_decision = -1
    if btc_to_sell < 0:
        trade_decision = 1
    if btc_to_sell == 0:
        return
    binance.render_trade_decision(
        symbol="BTCUSDT",
        quantity=abs(btc_to_sell),
        trade_decision=trade_decision
    )
    
def update_datastore_session(id, trade_decision, step_num):
    btc_price = get_btcusdt_price()
    current_bal = get_current_balance()
    account_value_hold = float(datastore.starting_usdt_balance)
    account_value_hold += float(get_btcusdt_price())
    btc_balance = get_balance_btc()
    usdt_balance = get_balance_usdt()
    # transaction_info = [
    #     {"btc_price": btc_price},
    #     {"trade_decision": trade_decision},
    #     {"account_value": current_bal},
    # ]
    datastore.create_transaction(
        step=step_num,
        transaction_type="trade",
        session_id=id,
        # btc_price=btc_price,
        # trade_decision=trade_decision,
        account_value_with_model=current_bal,
        account_value_hold_only=account_value_hold
        # btc_balance=btc_balance,
        # usdt_balance=usdt_balance
    )
    session = datastore.get_session_by_id(session_id=id)
    ending_balance = current_bal
    coins_bought = int(session.get("coins_bought"))
    coins_sold = int(session.get("coins_sold"))
    number_of_trades = int(session.get("number_of_trades"))
    ending_coins = get_balance_btc()
    if trade_decision == 1:
        coins_bought += 1
    if trade_decision == -1:
        coins_sold += 1
    if trade_decision != 0:
        number_of_trades += 1
    print(
        f"ending_balance: {ending_balance}\n"\
        f"coins_bought: {coins_bought}\n"\
        f"coins_sold: {coins_sold}\n"\
        f"number_of_trades: {number_of_trades}\n"\
        f"ending_coins: {ending_coins}\n"
    )
    datastore.edit_session(
        session_id=id,
        ending_balance=ending_balance,
        coins_bought=str(coins_bought),
        coins_sold=str(coins_sold),
        number_of_trades=str(number_of_trades),
        ending_coins=ending_coins
    )
    
def check_session_exists() -> bool:
    try:
        id = (datastore.get_session_by_id(SESSION_ID)).get("id")
    except:
        return False
    if id:
        return True
    return False

def get_prediction():
    rnn.get_trade_descision()

def get_current_balance() -> float:
    btc_bal = float(get_balance_btc())
    usdt_bal = float(get_balance_usdt())
    btc_exchange_rate = float(get_btcusdt_price())
    btc_usdt_value = btc_bal * btc_exchange_rate
    return usdt_bal + btc_usdt_value

def get_balance_usdt():
    balances = binance.get_account_balances(["USDT"])
    return balances[0].get("free")

def get_btcusdt_price():
    btc_info = binance.get_client().get_symbol_ticker(symbol="BTCUSDT")
    return btc_info.get("price")

def get_balance_btc():
    balances = binance.get_account_balances(["BTC"])
    return balances[0].get("free")

def get_total_steps(id):
    session = datastore.get_session_by_id(session_id=id)
    
def create_session():
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_balance = get_current_balance()
    starting_coins = get_balance_btc()
    datastore.starting_usdt_balance = get_balance_usdt()
    id = datastore.create_session(
        session_name=SESSION_NAME,
        type=TYPE,
        model_name=MODEL_NAME,
        crypto_type=CRYPTO_TYPE,
        session_start=current_datetime,
        starting_balance=current_balance,
        starting_coins=starting_coins,
        coins_bought=str(0),
        coins_sold=str(0),
        number_of_trades=str(0)
    )
    print(f"Session created. ID = {id}")
    print(
        f"Session Name: {SESSION_NAME}\n"\
        f"Type: {TYPE}\n"\
        f"Model Name: {MODEL_NAME}\n"\
        f"Crypto Type: {CRYPTO_TYPE}\n"\
        f"Session Start: {current_datetime}\n"\
        f"Starting Balance: {current_balance}\n"\
        f"Starting Coins: {starting_coins}"
    )
    return id

