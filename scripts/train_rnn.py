from rnn.rnn import Net
from torch import nn, optim

# Model hyperparameters
HIDDEN_SIZE = 100
INPUT_SIZE = 10
OUTPUT_SIZE = 10
NUM_LAYERS = 5
BATCH_SIZE = 1 
LR = 1e-5
EPOCHS = 50

def train_rnn():

    # build rnn model
    model = Net(
        input_size=INPUT_SIZE, 
        hidden_size=HIDDEN_SIZE, 
        output_size=OUTPUT_SIZE, 
        num_layers=NUM_LAYERS,
        batch_size=BATCH_SIZE
    )

    # train model
    model._train(
        epochs=EPOCHS, 
        lr=LR,
        optimizer=optim.Adam(model.parameters(), LR),
        criterion=nn.MSELoss(),
        sequence_length=48,
    )

    # validate model
    model.validate(sequence_length=48, batch_size=BATCH_SIZE)

    # save model
    model.save()
    
    model.clean_dataset_csvs(48)
