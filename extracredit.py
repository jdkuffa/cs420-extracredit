import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Constants
MAX_VOCAB = 10000
TRAIN_RATIO = 80
VAL_RATIO = 20

# File paths
SAMPLE_DATASET_PATH = "data/datasets/sample-pt.csv"
OUTPUT_TRAIN_PATH = "data/datasets/output_train.txt"
OUTPUT_VAL_PATH = "data/datasets/output_val.txt"
RESULTS_PATH = "results/rnn_training_results.csv"


class JavaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(JavaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        return self.fc(output)


def tokenize_java_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        java_code = f.read().split()
    return java_code


def convert_txt_to_sentences(file_path):
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            sentences.append(line.strip().split())
    return sentences


def train_rnn(model, train_loader, val_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    results = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = validate_rnn(model, val_loader, criterion, device)
        results.append((epoch, train_loss / len(train_loader), val_loss))
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    
    return results


def validate_rnn(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)


def main():
    if len(sys.argv) == 2:
        print("[DEBUG] Starting main()")
        
        corpus_path = sys.argv[1]
        print(f"[DEBUG] Corpus path: {corpus_path}")

        # Tokenize Java file
        vocab_tokens = tokenize_java_file(corpus_path)
        print(f"[DEBUG] Number of tokens in corpus: {len(vocab_tokens)}")

        # Create vocabulary
        vocab = {token: idx for idx, token in enumerate(set(vocab_tokens))}
        vocab_size = min(len(vocab), MAX_VOCAB)
        print(f"[DEBUG] Vocabulary size: {vocab_size}")

        # Prepare data
        train_data = convert_txt_to_sentences(OUTPUT_TRAIN_PATH)
        val_data = convert_txt_to_sentences(OUTPUT_VAL_PATH)
        print(f"[DEBUG] Number of training sentences: {len(train_data)}")
        print(f"[DEBUG] Number of validation sentences: {len(val_data)}")
        
        # Convert tokens to indices
        train_indices = [[vocab.get(token, vocab['<UNK>']) for token in seq] for seq in train_data]
        val_indices = [[vocab.get(token, vocab['<UNK>']) for token in seq] for seq in val_data]
        print(f"[DEBUG] Sample training indices: {train_indices[:1]}")
        print(f"[DEBUG] Sample validation indices: {val_indices[:1]}")

        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(torch.LongTensor(train_indices[:-1]), torch.LongTensor(train_indices[1:])), batch_size=32)
        val_loader = DataLoader(TensorDataset(torch.LongTensor(val_indices[:-1]), torch.LongTensor(val_indices[1:])), batch_size=32)
        print(f"[DEBUG] Training DataLoader created with {len(train_loader)} batches")
        print(f"[DEBUG] Validation DataLoader created with {len(val_loader)} batches")
        
        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DEBUG] Using device: {device}")
        model = JavaRNN(vocab_size, embedding_dim=128, hidden_dim=256).to(device)
        print("[DEBUG] Model initialized")

        # Train model
        results = train_rnn(model, train_loader, val_loader, epochs=10, device=device)
        print("[DEBUG] Training completed")

        # Save results
        with open(RESULTS_PATH, "w") as f:
            f.write("Epoch,Train Loss,Validation Loss\n")
            for epoch, train_loss, val_loss in results:
                f.write(f"{epoch+1},{train_loss},{val_loss}\n")
        print(f"[DEBUG] Results saved to {RESULTS_PATH}")

        print("Training completed. Results saved to results file.")


if __name__ == "__main__":
    main()