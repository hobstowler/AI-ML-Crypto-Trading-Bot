import pandas 
import numpy as np
import argparse

def generate_start_indicies(length, input_len, target_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):

    if abs((train_split + val_split + test_split) - 1.0) > 0.0001:
        raise ValueError("train_split + val_split + test_split must equal 1")

    if length < input_len + target_len:
        raise ValueError("length must be greater than input_len + target_len")
    
    set_length = input_len + target_len

    # Note: This will ignore the last set if it is not a full set
    num_sets = length // set_length

    if seed is not None:
        np.random.seed(seed)

    # Make a shuffle list of indicies of length num_sets
    shuffle_list = list(range(num_sets))
    np.random.shuffle(shuffle_list)

    train_start = 0
    val_start = int(train_split * num_sets)
    test_start = val_start + int(val_split * num_sets)

    train_indicies = np.multiply(shuffle_list[train_start:val_start], set_length)
    val_indicies = np.multiply(shuffle_list[val_start:test_start], set_length)
    test_indicies = np.multiply(shuffle_list[test_start:], set_length)

    return train_indicies, val_indicies, test_indicies



