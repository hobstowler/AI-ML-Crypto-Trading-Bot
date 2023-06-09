import pandas 
import torch
import numpy as np
import argparse
import os
#import sklearn
#print("Finished generate_datasets.py sklearn import")
import csv

def get_arguments():
    '''
    Function that collects cli arguments.
    '''
    parser = argparse.ArgumentParser(description = 'Generate a csv with dataset info')
    parser.add_argument('--input_file', help='Required. Input file to generate datasets from')
    parser.add_argument('--seq_len', help='Required. Length of individual sequences')
    parser.add_argument('--train_split', help='Optional. Percentage of data to use for training', default=0.8)
    parser.add_argument('--val_split', help='Optional. Percentage of data to use for validation', default=0.1)
    parser.add_argument('--test_split', help='Optional. Percentage of data to use for testing', default=0.1)
    parser.add_argument('--seed', help='Optional. Seed for random number generator', default=None)
    args = parser.parse_args()

    return args

def generate_start_indicies(total_len, seq_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):
    """
    Function that generates indicies for splitting data into train, val, and test sets. Does not do the actual 
    splitting, just generates the indicies.

    Args:
        length (int): Total length of the data to be processed.
        seq_len (int): Length of the individual sequences.
        train_split (float, optional): Percentage of data to use for training. Defaults to 0.8.
        val_split (float, optional): Percentage of data to use for validation. Defaults to 0.1.
        test_split (float, optional): Percentage of data to use for testing. Defaults to 0.1.
        seed (int, optional): Seed value for random shuffling. Use to make consistent data sets. Defaults to None.

    Raises:
        ValueError: train_split + val_split + test_split must equal 1
        ValueError: length must be greater than input_len + target_len

    Returns:
        tuple: tuple containing following lists: (training start indicies, validation start indicies, testing start indicies)
    """

    # Error checking on splits
    if abs((train_split + val_split + test_split) - 1.0) > 0.0001:
        raise ValueError("train_split + val_split + test_split must equal 1")

    # Note: This will ignore the last set if it is not a full set
    num_sets = total_len // seq_len

    if seed is not None:
        np.random.seed(seed)

    # Make a shuffle list of indicies of length num_sets
    shuffle_list = list(range(num_sets))
    np.random.shuffle(shuffle_list)

    train_start = 0
    val_start = int(train_split * num_sets)
    test_start = val_start + int(val_split * num_sets)

    train_indicies = np.multiply(shuffle_list[train_start:val_start], seq_len)
    val_indicies = np.multiply(shuffle_list[val_start:test_start], seq_len)
    test_indicies = np.multiply(shuffle_list[test_start:], seq_len)

    return train_indicies, val_indicies, test_indicies

def generate_full_indicies(total_len, seq_len, train_split=0.8, val_split=0.1, test_split=0.1, seed=None):
    '''
    Takes the start indicies and fills in the rest of the index values for each set for a given input or target length.
    Is not called directly by the user.
    '''

    train_indicies_start, val_indicies_start, test_indicies_start = generate_start_indicies(total_len, seq_len, train_split, val_split, test_split, seed)

    all_indicies = {}

    # Build arrays to hold indicies
    all_indicies['train_indicies'] = np.array([], dtype=np.int64)
    all_indicies['val_indicies'] = np.array([], dtype=np.int64)
    all_indicies['test_indicies'] = np.array([], dtype=np.int64)

    # Fill in training indicies
    for value in train_indicies_start:
        all_indicies['train_indicies'] = np.append(all_indicies['train_indicies'], np.arange(value, value + seq_len))

    # Fill in validation indicies
    for value in val_indicies_start:
        all_indicies['val_indicies'] = np.append(all_indicies['val_indicies'], np.arange(value, value + seq_len))

    # Fill in testing indicies
    for value in test_indicies_start:
        all_indicies['test_indicies'] = np.append(all_indicies['test_indicies'], np.arange(value, value + seq_len))

    return all_indicies

def get_data_types(df):
    '''
    Returns a list of the data types for each column in the dataframe.
    '''
    data_types = []
    for column in df.columns:
        data_types.append(df[column].dtype)

    return data_types

def normalize_data(df, scaler=None):
    '''
    Normalize every column that is a number type in the dataframe.
    '''

    # Generate list of dataframe columns whos type is a number
    number_columns = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            number_columns.append(column)

    # Normalize the data
    #scaler = sklearn.preprocessing.MinMaxScaler()

    # if params is not None:
    #     scaler.data_min_, scaler.data_max_, scaler.scale_ = params

    #df[number_columns] = scaler.fit_transform(df[number_columns])

    name = "test1"
    # Record the normalization values
    #record_normalization_values(scaler, f"normalization_values_{name}.csv", number_columns)

    return df, #scaler

def manual_normalization(df, min_max=None):
    '''
    Normalize every column that is a number type in the dataframe.
    '''

    # Generate list of dataframe columns whos type is a number
    number_columns = []
    for column in df.columns:
        if np.issubdtype(df[column].dtype, np.number):
            number_columns.append(column)
    
    # Build min_max dataframe if not provided
    if min_max is None:
        min_max = pandas.DataFrame(columns=number_columns)
        min_values = []
        max_values = []
        for column in number_columns:
            min_values.append(df[column].min())
            max_values.append(df[column].max())

        min_max.loc[0] = min_values
        min_max.loc[1] = max_values

        # Write min_max to file
        min_max.to_csv("min_max.csv")

    # Read min_max data and normalize the inputs
    min_max = pandas.read_csv("min_max.csv")
    for column in number_columns:
        df[column] = (df[column] - min_max.loc[0, column]) / (min_max.loc[1, column] - min_max.loc[0, column])

    # Normalize the data
    #df[number_columns] = (df[number_columns] - data_min) / (data_max - data_min)

    return df

def generate_csv_datasets(input_file, seq_len, train_split=0.8, val_split=0.1, test_split=0.1, norm_params=None, seed=None):
    """
    Generates the train, val and test datasets as normalized values for a csv file.

    Args:
        input_file (str): Path to the input csv file.
        seq_len (int): Length of the individual sequences.
        train_split (float, optional): Percentage of data to use for training. Defaults to 0.8.
        val_split (float, optional): Percentage of data to use for validation. Defaults to 0.1.
        test_split (float, optional): Percentage of data to use for testing. Defaults to 0.1.
        norm_params (tuple, optional): Tuple containing the normalization parameters. Defaults to None.
        seed (int, optional): Seed value for random shuffling. Use to make consistent data sets. Defaults to None.

    Returns:
        tuple: tuple containing following lists: (data_min, data_max, scale)
    """

    # Read in the input file
    df = pandas.read_csv(input_file)

    # Normalize the data
    df, scalar = normalize_data(df)


    all_indicies = generate_full_indicies(len(df), seq_len, train_split, val_split, test_split, seed)

    # Create the train, val, and test datasets
    train_df = df.iloc[all_indicies['train_indicies']]

    val_df = df.iloc[all_indicies['val_indicies']]

    test_df = df.iloc[all_indicies['test_indicies']]

    # Write the datasets to files
    train_df.to_csv(f"./train_{seq_len}.csv", index=False)
    val_df.to_csv(f"./val_{seq_len}.csv", index=False)
    test_df.to_csv(f"./test_{seq_len}.csv", index=False)

    return scalar

def tensors_from_csv(infile, seq_len, columns=[], batch_size=1):
    """
    Generates pytorch tensors from a csv file with the given sequence length and columns.

    Args:
        infile (str): Path to the input csv file.
        seq_len (int): Length of the sequences in the input file.
        columns (list): List of columns to use in the input file.
        batch_size (int, optional): Size of the batches to build. Defaults to 1.
    """

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

def record_normalization_values(scalar, output_file, number_columns):
    """
    Records the normalization values to a file.

    Args:
        scalar (MinMaxScaler): MinMaxScaler object containing the normalization values.
        output_file (str): Path to the output file.
    """

    with open(output_file, 'w') as f:
        write = csv.writer(f)
        write.writerow(number_columns)
        write.writerow(scalar.data_min_)
        write.writerow(scalar.data_max_)
        write.writerow(scalar.scale_)

def denormalize_data(predictions, normalization_file, column_names):
    """
    De-normalizes the predictions given the normalization values.

    Args:
        predictions (numpy array): Array of predictions to de-normalize.
        normalization_file (path): File holding the normalization values.
        column_names (list): List of column names to use in the normalization file.

    Returns:
        numpy array: De-normalized predictions.
    """

    df = pandas.read_csv(normalization_file)
    df = df[column_names]

    data_min = df.iloc[0].to_numpy()
    data_max = df.iloc[1].to_numpy()

    predictions = predictions.detach().numpy()

    predictions = predictions * (data_max - data_min) + data_min

    return predictions

def inference_df_to_tensor(df, seq_len, columns=[]):

    tensor = np.zeros((1, seq_len, len(columns)))
    tensor[0] = df[columns].to_numpy()
    torch_tensor = torch.from_numpy(tensor)

    return torch_tensor


def clean_dataset_csv_files(seq_len):
    os.remove(f"./train_{seq_len}.csv")
    os.remove(f"./val_{seq_len}.csv")
    os.remove(f"./test_{seq_len}.csv")
    
    
if __name__ == '__main__':

    # Get arguments from the command line
    args = get_arguments()
    # Call generate_csv_datasets with the cli arguments
    generate_csv_datasets(args.input_file, int(args.seq_len), float(args.train_split), 
                           float(args.val_split), float(args.test_split), args.seed)
    

