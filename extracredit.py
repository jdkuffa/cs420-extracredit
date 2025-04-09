import os
import re
import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import javalang


# Constants for configuration and file paths
DATASET_FILE_PATH = "data/sample-pt.csv"  # Replace with the actual file path
VOCAB_SIZE = 10000  # Maximum vocabulary size
EMBEDDING_DIM = 128  # Dimension of word embeddings
HIDDEN_DIM = 256  # Dimension of RNN hidden state
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 10  # Number of training epochs
OUTPUT_CSV_FILE = "results/training_performance.csv"  # Path to save performance log


# Step 1: Clean Java Code to Remove Invalid Characters
def clean_java_code(java_code):
    """
    Cleans Java code by:
    - Removing non-ASCII characters.
    - Replacing non-breaking spaces with regular spaces.
    - Removing or escaping problematic characters.
    """
    java_code = java_code.replace('\xa0', ' ')  # Replace non-breaking spaces with regular spaces.
    java_code = re.sub(r'[^\x00-\x7F]+', '', java_code)  # Remove non-ASCII characters.
    java_code = re.sub(r'\\', r'\\\\', java_code)  # Escape backslashes.
    return java_code.strip()

# Step 2: Tokenize Java Code
def tokenize_java_code(java_code):
    try:
        tokens = list(javalang.tokenizer.tokenize(java_code))
        return [token.value for token in tokens]
    except javalang.tokenizer.LexerError as e:
        print(f"Skipping invalid code due to LexerError: {e}")
        return []  # Return an empty list if tokenization fails.

# Step 3: Load, Clean, and Tokenize Dataset
def load_and_tokenize_dataset(file_path):
    with open(file_path, 'r') as file:
        java_methods = file.readlines()
    
    cleaned_methods = [clean_java_code(method) for method in java_methods]
    tokenized_methods = [tokenize_java_code(method) for method in cleaned_methods if method.strip()]
    return tokenized_methods

# Step 4: Mask Tokens (15% masking)
def mask_tokens(tokenized_methods, mask_probability=0.15):
    masked_data = []
    for method in tokenized_methods:
        method_length = len(method)
        num_masks = int(method_length * mask_probability)
        mask_indices = random.sample(range(method_length), num_masks)
        
        for i, index in enumerate(mask_indices):
            prepared_input = method.copy()
            prepared_input[index] = f'MASK_{i+1}'
            masked_data.append({
                'original_input': method,
                'prepared_input': prepared_input,
                'output': method[index]
            })
    return masked_data

# Step 5: Build Vocabulary (Cap at VOCAB_SIZE tokens)
def build_vocabulary(tokenized_methods, vocab_size):
    all_tokens = [token for method in tokenized_methods for token in method]
    most_common_tokens = [token for token, _ in Counter(all_tokens).most_common(vocab_size)]
    vocab = {token: idx for idx, token in enumerate(most_common_tokens)}
    vocab['<UNK>'] = len(vocab)  # Add unknown token
    return vocab

# Step 6: Encode Data
def encode_data(masked_data, vocab):
    encoded_data = []
    for instance in masked_data:
        encoded_instance = {
            'original_input': [vocab.get(token, vocab['<UNK>']) for token in instance['original_input']],
            'prepared_input': [vocab.get(token, vocab['<UNK>']) for token in instance['prepared_input']],
            'output': vocab.get(instance['output'], vocab['<UNK>'])
        }
        encoded_data.append(encoded_instance)
    return encoded_data

# Step 7: Custom Dataset Class
class JavaDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'prepared_input': torch.tensor(self.data[idx]['prepared_input'], dtype=torch.long),
            'output': torch.tensor(self.data[idx]['output'], dtype=torch.long)
        }

# Step 8: Define RNN Model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.fc(output[:, -1])  # Predict only the last token (masked one)
        return output

# Step 9: Train Model
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    performance_log = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch in dataloader:
            inputs = batch['prepared_input']
            targets = batch['output']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        performance_log.append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')
    
    return performance_log

# Step 10: Save Performance Log to CSV
def save_performance_log(performance_log, output_file):
    df = pd.DataFrame(performance_log)
    df.to_csv(output_file, index=False)

# Step 11: Plot Training Loss
def plot_training_loss(performance_log):
    epochs = [entry['epoch'] for entry in performance_log]
    losses = [entry['loss'] for entry in performance_log]
    
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.show()

# Main Script Execution
if __name__ == "__main__":
    sample = pd.read_csv(DATASET_FILE_PATH)
    print(sample.loc[0])
    
    # Load and preprocess dataset
    print("Loading and preprocessing dataset...")
    tokenized_methods = load_and_tokenize_dataset(DATASET_FILE_PATH)
    
    print("Masking tokens...")
    masked_data = mask_tokens(tokenized_methods)

    print("Building vocabulary...")
    vocab = build_vocabulary(tokenized_methods, VOCAB_SIZE)

    print("Encoding data...")
    encoded_data = encode_data(masked_data, vocab)

    # Create DataLoader
    print("Creating DataLoader...")
    dataset = JavaDataset(encoded_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model and training components
    print("Initializing model...")
    model = RNNModel(vocab_size=len(vocab), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train model
    print("Training model...")
    performance_log = train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS)

    # Save performance log to CSV
    print("Saving performance log...")
    save_performance_log(performance_log, OUTPUT_CSV_FILE)

    # Plot training loss
    print("Plotting training loss...")
    plot_training_loss(performance_log)

    print("Execution complete.")
