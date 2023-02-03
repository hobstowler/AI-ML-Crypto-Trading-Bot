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


def generate_full_indicies(length, input_len, target_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):

    train_indicies_start, val_indicies_start, test_indicies_start = generate_start_indicies(length, input_len, target_len, train_split, val_split, test_split, seed)

    all_indicies = {}

    all_indicies['train_inputs'] = np.array([], dtype=np.int64)
    all_indicies['train_targets'] = np.array([], dtype=np.int64)
    all_indicies['val_inputs'] = np.array([], dtype=np.int64)
    all_indicies['val_targets'] = np.array([], dtype=np.int64)
    all_indicies['test_inputs'] = np.array([], dtype=np.int64)
    all_indicies['test_targets'] = np.array([], dtype=np.int64)

    for value in train_indicies_start:
        all_indicies['train_inputs'] = np.append(all_indicies['train_inputs'], np.arange(value, value + input_len))
        all_indicies['train_targets'] = np.append(all_indicies['train_targets'], np.arange(value + input_len, value + input_len + target_len))

    for value in val_indicies_start:
        all_indicies['val_inputs'] = np.append(all_indicies['val_inputs'], np.arange(value, value + input_len))
        all_indicies['val_targets'] = np.append(all_indicies['val_targets'], np.arange(value + input_len, value + input_len + target_len))

    for value in test_indicies_start:
        all_indicies['test_inputs'] = np.append(all_indicies['test_inputs'], np.arange(value, value + input_len))
        all_indicies['test_targets'] = np.append(all_indicies['test_targets'], np.arange(value + input_len, value + input_len + target_len))

    return all_indicies


def generate_datasets(input_file, input_len, target_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):

    # Read in the input file
    df = pandas.read_csv(input_file)

    all_indicies = generate_full_indicies(len(df), input_len, target_len, train_split, val_split, test_split, seed)

    # Create the train, val, and test datasets
    train_input_df = df.iloc[all_indicies['train_inputs']]
    train_target_df = df.iloc[all_indicies['train_targets']]

    val_input_df = df.iloc[all_indicies['val_inputs']]
    val_target_df = df.iloc[all_indicies['val_targets']]

    test_input_df = df.iloc[all_indicies['test_inputs']]
    test_target_df = df.iloc[all_indicies['test_targets']]

    # Write the datasets to files
    train_input_df.to_csv(f"./train_input_{input_len}.csv", index=False)
    train_target_df.to_csv(f"./train_target_{target_len}.csv", index=False)
    val_input_df.to_csv(f"./val_input_{input_len}.csv", index=False)
    val_target_df.to_csv(f"./val_target_{target_len}.csv", index=False)
    test_input_df.to_csv(f"./test_input_{input_len}.csv", index=False)
    test_target_df.to_csv(f"./test_target_{target_len}.csv", index=False)

#generate_datasets("./data_conversion/test_data.csv", 3, 1, 0.8, 0.1, 0.1, 0)


