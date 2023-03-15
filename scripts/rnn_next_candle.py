from rnn.rnn import Net

MODEL_NAME = "alpha.rnn"
MODEL_PATH = f"rnn/saved_rnn_models/{MODEL_NAME}"

HIDDEN_SIZE = 100
INPUT_SIZE = 10
OUTPUT_SIZE = 10
NUM_LAYERS = 5
BATCH_SIZE = 1 

def rnn_next_candle():
    rnn = Net(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        output_size=OUTPUT_SIZE, 
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE
    )
    rnn.load(MODEL_PATH)
    price = rnn.get_next_close_price()
    print("\n---- NEXT CLOSE PRICE ----")
    print(price)
