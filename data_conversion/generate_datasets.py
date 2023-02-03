import pandas 
import torch
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler

def generate_start_indicies(length, input_len, target_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):
    '''
    All this function does is caculate where the start indicies of each set are. 
    It does the train/val/test split and shuffles the start indicies.
    '''

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
    '''
    Takes the start indicies and fills in the rest of the index values for each set for a given input or target length.
    '''

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

def normalize_data(df):
    '''
    Normalize every column that is a number type in the dataframe
    '''

    # Generate list of dataframe columns whos type is a number
    number_columns = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            number_columns.append(column)

    # Normalize the data
    scaler = MinMaxScaler()
    df[number_columns] = scaler.fit_transform(df[number_columns])
    return df

def generate_csv_datasets(input_file, input_len, target_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):
    '''
    Makes the train, val, and test datasets from the input file and returns them as csv files.
    '''

    # Read in the input file
    df = pandas.read_csv(input_file)

    # Normalize the data
    df = normalize_data(df)

    all_indicies = generate_full_indicies(len(df), input_len, target_len, train_split, val_split, test_split, seed)

    # Create the train, val, and test datasets
    train_input_df = df.iloc[all_indicies['train_inputs']]
    train_target_df = df.iloc[all_indicies['train_targets']]

    val_input_df = df.iloc[all_indicies['val_inputs']]
    val_target_df = df.iloc[all_indicies['val_targets']]

    test_input_df = df.iloc[all_indicies['test_inputs']]
    test_target_df = df.iloc[all_indicies['test_targets']]

    folder_name='data_conversion'

    # Write the datasets to files
    train_input_df.to_csv(f"./{folder_name}/train_input_{input_len}.csv", index=False)
    train_target_df.to_csv(f"./{folder_name}/train_target_{target_len}.csv", index=False)
    val_input_df.to_csv(f"./{folder_name}/val_input_{input_len}.csv", index=False)
    val_target_df.to_csv(f"./{folder_name}/val_target_{target_len}.csv", index=False)
    test_input_df.to_csv(f"./{folder_name}/test_input_{input_len}.csv", index=False)
    test_target_df.to_csv(f"./{folder_name}/test_target_{target_len}.csv", index=False)

def tensors_from_csv(infile, seq_len, columns=[], batch_size=1):
    '''
    Makes pytorch tensors of shape (batch_size, seq_len, num_features) from a dataframe.
    '''

    # Read in the input file
    df = pandas.read_csv(infile)

    tensors = []
    num_sequences = len(df) // seq_len
    num_tensors = num_sequences // batch_size

    # Make each individual tensor
    for tensor_idx in range(num_tensors):

        tensor = np.zeros((batch_size, seq_len, len(columns)))

        for batch in range(batch_size):
                
            batch_start = (tensor_idx * batch_size + batch) * seq_len
            batch_end = (tensor_idx * batch_size + batch + 1) * seq_len
            batch_df = df.iloc[batch_start:batch_end]
            batch_df = batch_df[columns]
            tensor[batch] = batch_df.to_numpy()

        tensors.append(torch.from_numpy(tensor))

    return tensors

# generate_csv_datasets("./data_conversion/2021_hourly.csv", input_len=48, target_len=12, train_split=0.8, val_split=0.1, test_split=0.1, seed=0)
# train_target_tensors = tensors_from_csv("./data_conversion/train_target_12.csv", seq_len=12, columns=['close_price', 'volume'], batch_size=1)
# print(len(train_target_tensors))
# print(train_target_tensors[0].shape)
# print(train_target_tensors[0])


