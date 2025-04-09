from collections import Counter
import torch
import torch.nn as nn
import pandas as pd
import random
import os


# Constants
VOCAB_SIZE = 10000  # Maximum vocabulary size

# File paths
SAMPLE_DATA_FILE = 'datasets/data/sample-pt.csv'
INPUT_TRAIN_PATH = 'datasets/generated_files/input_train_dataset.csv'
INPUT_EVAL_PATH = 'datasets/generated_files/input_eval_dataset.csv'
INPUT_TEST_PATH = 'datasets/generated_files/input_test_dataset.csv'
TARGET_TRAIN_PATH = 'datasets/generated_files/target_train_dataset.csv'
TARGET_EVAL_PATH = 'datasets/generated_files/target_eval_dataset.csv'
TARGET_TEST_PATH = 'datasets/generated_files/target_test_dataset.csv'


# Class for simple LSTM-based model
class LSTMModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
		super(LSTMModel, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		if isinstance(x, int):  # If x is an integer, convert it
			x = torch.tensor([x], dtype=torch.long)  # Convert to tensor

		embedded = self.embedding(x)
		lstm_out, (hidden, cell) = self.lstm(embedded)
		logits = self.fc(lstm_out)
		return logits, hidden


def load_dataset(file_path):
    """Load the dataset from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file at {file_path} is empty or malformed. Please provide a valid CSV file.")


def split_dataset(data, train_ratio=0.6, eval_ratio=0.2):
    """Split the dataset into training, evaluation, and testing sets."""
    total_size = len(data)
    indices = list(range(total_size))
    random.shuffle(indices)  # Shuffle the indices for randomization

    train_size = int(train_ratio * total_size)
    eval_size = int(eval_ratio * total_size)

    train_indices = indices[:train_size]
    eval_indices = indices[train_size:train_size + eval_size]
    test_indices = indices[train_size + eval_size:]

    train_data = data.iloc[train_indices]
    eval_data = data.iloc[eval_indices]
    test_data = data.iloc[test_indices]

    return train_data, eval_data, test_data


def save_to_csv(data, column_name, file_name):
    """Save a specific column of the dataset to a CSV file."""
    pd.DataFrame({column_name: data[column_name]}).to_csv(file_name, index=False)


def save_datasets(train_data, eval_data, test_data):
    """Save datasets to CSV files after splitting."""

    # Check and save the input datasets
    if not os.path.exists(INPUT_TRAIN_PATH):
        save_to_csv(train_data, "prepared_input", INPUT_TRAIN_PATH)
    if not os.path.exists(INPUT_EVAL_PATH):
        save_to_csv(eval_data, "prepared_input", INPUT_EVAL_PATH)
    if not os.path.exists(INPUT_TEST_PATH):
        save_to_csv(test_data, "prepared_input", INPUT_TEST_PATH)

    # Check and save the target datasets
    if not os.path.exists(TARGET_TRAIN_PATH):
        save_to_csv(train_data, "output", TARGET_TRAIN_PATH)
    if not os.path.exists(TARGET_EVAL_PATH):
        save_to_csv(eval_data, "output", TARGET_EVAL_PATH)
    if not os.path.exists(TARGET_TEST_PATH):
        save_to_csv(test_data, "output", TARGET_TEST_PATH)


def tokenize_dataset(methods):
    """Tokenize the dataset."""
    token_list = [token for seq in methods for token in seq.split()]

    # Add special tokens
    token_list.append("[COMPLETION_N]")  # Ensure one occurrence of [COMPLETION_N]
    token_list.append("<PAD>")  # Ensure padding token is included

    # Count the frequency of each token
    token_counts = Counter(token_list)

    # Select the most common tokens up to the specified vocabulary size
    most_common_tokens = [token for token, _ in token_counts.most_common(VOCAB_SIZE - 2)]

    # Add special tokens
    most_common_tokens.append("[COMPLETION_N]")  # Ensure one occurrence of [COMPLETION_N]
    most_common_tokens.append("<PAD>")  # Ensure padding token is included

    return most_common_tokens


def load_sample_data():
    return load_dataset(SAMPLE_DATA_FILE)


def tokenize_dataset(methods):
    """Tokenize the dataset."""
    # Extract unique tokens (words) from the methods
    unique_tokens = set(token for seq in methods for token in seq.split())

    # Add special tokens
    unique_tokens.add("[COMPLETION_N]")  # Ensure one occurrence of [COMPLETION_N]
    unique_tokens.add("<PAD>")  # Ensure padding token is included

    # Count the frequency of each token
    token_counts = Counter(unique_tokens)

    # Select the most common tokens up to the specified vocabulary size
    most_common_tokens = [token for token, _ in token_counts.most_common(VOCAB_SIZE - 2)]

    # Add special tokens
    most_common_tokens.append("[COMPLETION_N]")  # Ensure one occurrence of [COMPLETION_N]
    most_common_tokens.append("<PAD>")  # Ensure padding token is included

    return most_common_tokens


def create_vocabulary(methods):
    most_common_tokens = tokenize_dataset(methods)

    # Assign each token a unique index
    vocab = {token: idx for idx, token in enumerate(most_common_tokens)}

    # Create bidirectional mappings 
    token_to_id = vocab
    id_to_token = {idx: token for token, idx in vocab.items()}

    print("\nVocabulary size:", len(token_to_id))
    print("'<PAD>' token ID:", token_to_id['<PAD>'])
    print("5 random vocabulary items:", list(token_to_id.items())[:5])

    return token_to_id, id_to_token


def split_and_save_datasets(sample_data):
    # Split the dataset into training, evaluation, and testing sets
    train_data, eval_data, test_data = split_dataset(sample_data)
    
    # Save the datasets to CSV files
    save_datasets(train_data, eval_data, test_data)

    return train_data, eval_data, test_data


def load_datasets():
    # Load the datasets from CSV files
    input_train_dataset = load_dataset(INPUT_TRAIN_PATH)
    target_train_dataset = load_dataset(TARGET_TRAIN_PATH)

    input_eval_dataset = load_dataset(INPUT_EVAL_PATH)
    target_eval_dataset = load_dataset(TARGET_EVAL_PATH)

    input_test_dataset = load_dataset(INPUT_TEST_PATH)
    target_test_dataset = load_dataset(TARGET_TEST_PATH)
    
    return  input_train_dataset, target_train_dataset, \
            input_eval_dataset, target_eval_dataset,   \
            input_test_dataset, target_test_dataset


def get_max_sequence_length(*datasets):
    """Calculate maximum length across all datasets"""
    max_len = 0
    for data in datasets:
        for seq in data:
            max_len = max(max_len, len(seq))
    return max_len


def calculate_max_sequence_length(input_train_dataset, target_train_dataset, 
                                  input_eval_dataset, target_eval_dataset, 
                                  input_test_dataset, target_test_dataset):
    # Calculate maximum sequence length across all datasets
    max_seq_length = get_max_sequence_length(
        input_train_dataset["prepared_input"], target_train_dataset["output"],
        input_eval_dataset["prepared_input"], target_eval_dataset["output"],
        input_test_dataset["prepared_input"], target_test_dataset["output"]
    )

    return max_seq_length


def process_first_sample(input_train_dataset, target_train_dataset, token_to_id, id_to_token, max_seq_length):
    # Process the first sample from the training dataset
    SEQ_LENGTH = max_seq_length
    sample_input = input_train_dataset.iloc[0]
    sample_target = target_train_dataset.iloc[0]
    
    print("Original input:", sample_input["prepared_input"])
    print("Original target:", sample_target["output"])
    
    # Convert tokens to IDs
    input_ids = [token_to_id.get(token, token_to_id['<PAD>']) for token in sample_input["prepared_input"].split()]
    target_ids = [token_to_id.get(token, token_to_id['<PAD>']) for token in sample_target["output"].split()]
    
    if not all(isinstance(id, int) for id in input_ids):
        raise ValueError("input_ids must be a list of integers. Ensure the input data is tokenized and converted to IDs.")
    if not all(isinstance(id, int) for id in target_ids):
        raise ValueError("target_ids must be a list of integers. Ensure the target data is tokenized and converted to IDs.")
    
    # Manual padding
    padded_input = input_ids[:SEQ_LENGTH] + [token_to_id['<PAD>']] * (SEQ_LENGTH - len(input_ids))
    padded_target = target_ids[:SEQ_LENGTH] + [token_to_id['<PAD>']] * (SEQ_LENGTH - len(target_ids))
    
    print("\nOriginal input length:", len(sample_input))
    print("Padded input length:", len(padded_input))
    print("Input before/after padding:")
    print("Original Input:\n", sample_input)
    print("Padded Input:", [id_to_token[id] for id in padded_input])
    print("Padded Target:", [id_to_token[id] for id in padded_target])


def main():
    print("Starting main process...")
    sample_data = load_sample_data()
    print("Sample Data: \n", sample_data.head())

    print("\nCreating vocabulary...")
    token_to_id, id_to_token = create_vocabulary(sample_data)

    print("\nVocabulary size:", len(token_to_id))
    print("'<PAD>' token ID:", token_to_id['<PAD>'])
    print("5 random vocabulary items:", list(token_to_id.itemgits())[:5])
    
    print("\nSplitting and saving datasets...")
    train_data, eval_data, test_data = split_and_save_datasets(sample_data)
    
    print("\nLoading datasets...")
    input_train_dataset, target_train_dataset, input_eval_dataset, target_eval_dataset, input_test_dataset, target_test_dataset = load_datasets()
    
    print("\nCalculating maximum sequence length...")
    max_seq_length = calculate_max_sequence_length(input_train_dataset, target_train_dataset, input_eval_dataset, target_eval_dataset, input_test_dataset, target_test_dataset)
    print("Maximum sequence length:", max_seq_length)

    print("\nProcessing first sample...")
    process_first_sample(input_train_dataset, target_train_dataset, token_to_id, id_to_token, max_seq_length)


if __name__ == "__main__":
    main()