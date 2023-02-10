import os
import torch

def save_model(model, optimizer, save_path):
    # check if the models folder exists and if not make it
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)

