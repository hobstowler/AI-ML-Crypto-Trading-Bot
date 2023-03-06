from rnn.rnn import Net

def rnn_next_candle():
    rnn = Net()
    rnn.load_alpha()
    price = rnn.get_next_close_price()
    print("\n---- NEXT CLOSE PRICE ----")
    print(price)
