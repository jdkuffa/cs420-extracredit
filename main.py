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
    return pd.read_csv(file_path)

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

def get_max_sequence_length(*datasets):
    """Calculate maximum length across all datasets."""
    max_len = 0
    for data in datasets:
        for seq in data:
            max_len = max(max_len, len(seq))
    return max_len


def main():
    # Load and inspect the sample data
    print("Loading and inspecting sample data...")
    sample_data = load_dataset(SAMPLE_DATA_FILE)
    print("Sample Data: \n", sample_data.head())

    # Extract input methods from the relevant columns
    methods = sample_data["prepared_input"].tolist()  # Convert the column to a list of strings

    # Tokenize the dataset and create a vocabulary from the most common tokens
    print("\nTokenizing dataset and creating vocabulary...")
    most_common_tokens = tokenize_dataset(methods)

    # Assign each token a unique index
    vocab = {token: idx for idx, token in enumerate(most_common_tokens)}

    # Create bidirectional mappings 
    token_to_id = vocab
    id_to_token = {idx: token for token, idx in vocab.items()}
    print("\nVocabulary size:", len(token_to_id))
    print("'<PAD>' token ID:", token_to_id['<PAD>'])
    print("5 random vocabulary items:", list(token_to_id.items())[:5])

    # Randomly split the dataset
    print("\nSplitting datasets into train, eval, and test sets...")
    train_data, eval_data, test_data = split_dataset(sample_data)

    # Save datasets
    print("Saving datasets to CSV files...")
    save_datasets(train_data, eval_data, test_data)
    print("\nDatasets split and saved successfully.")

    # Load the datasets
    print("\nLoading input and target datasets from CSV files...")
    input_train_dataset = load_dataset(INPUT_TRAIN_PATH)
    target_train_dataset = load_dataset(TARGET_TRAIN_PATH)

    input_eval_dataset = load_dataset(INPUT_EVAL_PATH)
    target_eval_dataset = load_dataset(TARGET_EVAL_PATH)
    
    input_test_dataset = load_dataset(INPUT_TEST_PATH)
    target_test_dataset = load_dataset(TARGET_TEST_PATH)
 
    # Manually preprocess the data by padding sequences in input and target datasets
    print("\nCalculating maximum sequence length across datasets...")
    max_seq_length = get_max_sequence_length(
        input_train_dataset, target_train_dataset,
        input_eval_dataset, target_eval_dataset,
        input_test_dataset, target_test_dataset
    )

    print(f"Maximum sequence length: {max_seq_length}")

    # Process first sample without using Dataset class
    print("\nProcessing first sample without using Dataset class...")
    SEQ_LENGTH = max_seq_length
    sample_input = input_train_dataset.iloc[0]
    sample_target = target_train_dataset.iloc[0]
    print("Original input:", sample_input["prepared_input"])        
    print("Original target:", sample_target["output"])

    # Convert tokens to IDs
    input_ids = [token_to_id.get(token, token_to_id['<PAD>']) for token in sample_input]
    target_ids = [token_to_id.get(token, token_to_id['<PAD>']) for token in sample_target]

    # Manual padding
    padded_input = input_ids[:SEQ_LENGTH] + [token_to_id['<PAD>']] * (SEQ_LENGTH - len(input_ids))
    padded_target = target_ids[:SEQ_LENGTH] + [token_to_id['<PAD>']] * (SEQ_LENGTH - len(target_ids))
    
    print("\nOriginal input length:", len(sample_input))
    print("Padded input length:", len(padded_input))
    print("Input before/after padding:")
    print("Original Input:", sample_input)
    print("Padded Input:", [id_to_token[id] for id in padded_input])


if __name__ == "__main__":
    main()