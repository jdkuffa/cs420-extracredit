# Import necessary libraries
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from itertools import product
from tqdm import tqdm
import numpy as np
import sys
from transformers import AutoModelForCausalLM

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Constants
MAX_VOCAB_SIZE = 10000  # Maximum vocabulary size

# File paths
SAMPLE_DATA_FILE = 'datasets/data/sample-pt.csv'
INPUT_TRAIN_PATH = 'datasets/generated_files/input_train_dataset.csv'
INPUT_EVAL_PATH = 'datasets/generated_files/input_eval_dataset.csv'
INPUT_TEST_PATH = 'datasets/generated_files/input_test_dataset.csv'
TARGET_TRAIN_PATH = 'datasets/generated_files/target_train_dataset.csv'
TARGET_EVAL_PATH = 'datasets/generated_files/target_eval_dataset.csv'
TARGET_TEST_PATH = 'datasets/generated_files/target_test_dataset.csv'


# Custom Dataset class
class CodeCompletionDataset(Dataset):
    def __init__(self, input_data, target_data, vocab, seq_length):
        """
        Custom dataset for code completion.

        Args:
            input_data (list of list of str): Tokenized input sequences.
            target_data (list of list of str): Corresponding tokenized target sequences.
            vocab (dict): Token-to-index mapping.
            seq_length (int): Fixed sequence length for padding.
        """
        self.input_data = input_data
        self.target_data = target_data
        self.vocab = vocab
        self.seq_length = seq_length

        # Ensure <PAD> token exists in vocabulary
        if '<PAD>' not in vocab:
            raise ValueError("Vocabulary must contain '<PAD>' token for padding.")

        self.pad_token_id = vocab['<PAD>']

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx]
        target_seq = self.target_data[idx]

        # Convert tokens to indices
        input_seq = [self.vocab.get(token, self.pad_token_id) for token in input_seq]
        target_seq = [self.vocab.get(token, self.pad_token_id) for token in target_seq]

        # Pad sequences to the fixed length
        input_seq = self.pad_sequence(input_seq, self.seq_length)
        target_seq = self.pad_sequence(target_seq, self.seq_length)

        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

    def pad_sequence(self, sequence, max_len):
        """
        Pads or truncates a sequence to a fixed length.

        Args:
            sequence (list of int): List of token indices.
            max_len (int): Desired sequence length.

        Returns:
            list of int: Padded/truncated sequence.
        """
        return sequence[:max_len] + [self.pad_token_id] * max(0, max_len - len(sequence))


# Model Architectures
class VanillaRNN(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout=0):
		super(VanillaRNN, self).__init__()
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
		self.fc = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		if isinstance(x, int):  # If x is an integer, convert it
			x = torch.tensor([x], dtype=torch.long)  # Convert to tensor

		embedded = self.embedding(x)
		output, hidden = self.rnn(embedded)
		out = self.fc(output)
		return out, hidden


# Define simple LSTM-based model
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


# Training functions
def train_model(model, dataloader, optimizer, criterion, epochs):
	"""Train the model and display step-wise training details."""
	model.train()
	display_interval = 100

	for epoch in range(epochs):
		epoch_loss = 0
		total_correct = 0
		total_samples = 0

		progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")

		for batch_idx, (inputs, targets) in progress_bar:
			optimizer.zero_grad()

			inputs, targets = inputs.to(device), targets.to(device)

			outputs, _ = model(inputs)

			# Compute loss
			loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
			loss.backward()
			optimizer.step()

			# Compute accuracy (if applicable)
			predictions = outputs.argmax(dim=-1)
			correct = (predictions == targets).sum().item()
			total = targets.numel()

			# Occasionally print input, target, and prediction for interpretation
			if batch_idx % display_interval == 0:
				print(f"Sample {idx}:")
				print(f"Input: {inputs[0].cpu().numpy()}")
				print(f"Target: {targets[0].cpu().numpy()}")
				print(f"Prediction: {predicted_tokens[0].cpu().numpy()}\n")

			batch_accuracy = correct / total
			total_correct += correct
			total_samples += total

			epoch_loss += loss.item()

			# Update progress bar with loss and accuracy
			progress_bar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_accuracy:.2%}")

		# Compute epoch-level metrics
		avg_loss = epoch_loss / len(dataloader)
		avg_accuracy = total_correct / total_samples

		print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2%}")

	# Save the trained model
	torch.save(model.state_dict(), "models/dl_code_completion.pth")
	print("Model saved successfully as rnn_code_completion.pth")


# 5. Utilities and Helper Functions
# Function to generate a vocabulary with a single occurrence of [COMPLETION_N]
def generate_vocab_with_completion_n(methods):
	"""
	Generates a vocabulary dictionary from a list of tokenized methods,
	ensuring that '[COMPLETION_N]' appears exactly once.

	Args:
		methods (list of list of str): Tokenized Java methods.

	Returns:
		dict: Vocabulary mapping each unique token to a unique index.
	"""
	# Count the frequency of each token
	token_counts = Counter(token for seq in methods for token in seq)

	# Add special tokens
	special_tokens = ["<PAD>", "[COMPLETION_N]", "<UNK>"]

	# Take the most common tokens excluding the special tokens
	most_common_tokens = [token for token, _ in token_counts.most_common(MAX_VOCAB_SIZE - len(special_tokens))]

	# Include special tokens first
	vocab_tokens = special_tokens + most_common_tokens

	# Assign each token a unique index
	vocab = {token: idx for idx, token in enumerate(vocab_tokens)}

	return vocab    

# 6. Hyperparameter Tuning
class HyperparameterTuner:
	def __init__(self, model_class, param_grid, train_dataloader, eval_dataloader, vocab_size, vocab):
		"""
		Initialize the hyperparameter tuner.

		Args:
			model_class: The model class to instantiate.
			param_grid: Dictionary of hyperparameters to search over.
			train_dataloader: DataLoader for training data.
			eval_dataloader: DataLoader for evaluation data.
			vocab_size: Size of the vocabulary for model initialization.
		"""
		self.model_class = model_class
		self.param_grid = param_grid
		self.train_dataloader = train_dataloader
		self.eval_dataloader = eval_dataloader
		self.vocab_size = vocab_size
		self.vocab = vocab

	def train_model(self, model, optimizer, criterion, epochs):
		"""Train the model and display step-wise training details."""
		model.train()
		display_interval = 100
		for epoch in range(epochs):
			epoch_loss = 0
			total_correct = 0
			total_samples = 0

			progress_bar = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc=f"Epoch {epoch+1}")

			for batch_idx, (inputs, targets) in progress_bar:

				inputs, targets = inputs.to(device), targets.to(device)

				optimizer.zero_grad()
				outputs,_ = model(inputs)

				# Compute loss
				loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), 2)
				optimizer.step()

				# Compute accuracy (if applicable)
				predictions = outputs.argmax(dim=-1)
				correct = (predictions == targets).sum().item()
				predicted_tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in predictions[5].tolist()]
				input_tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in inputs[5].cpu().numpy().tolist()]
				target_tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in targets[5].cpu().numpy().tolist()]

				total = targets.numel()

				# Occasionally print input, target, and prediction for interpretation
				if batch_idx % display_interval == 0:
					print(f"Sample {batch_idx}:")
					#print("******************************\n\n\n")
					#print(predictions[0:3])
					#print(inputs[0:3])
					#print(targets[0:3])
					#print("******************************\n\n\n")

					print(f"Input: {' '.join(input_tokens)}")
					print(f"Target: {' '.join(target_tokens)}")
					print(f"Prediction: {' '.join(predicted_tokens)}\n")
					#sys.exit(-1)

				batch_accuracy = correct / total
				total_correct += correct
				total_samples += total

				epoch_loss += loss.item()

				# Update progress bar with loss and accuracy
				progress_bar.set_postfix(loss=f"{loss.item():.4f}", accuracy=f"{batch_accuracy:.2%}")

			# Compute epoch-level metrics
			avg_loss = epoch_loss / len(self.train_dataloader)
			avg_accuracy = total_correct / total_samples

			print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.2%}")

	def evaluate_model(self, model, criterion):
		"""Evaluate the model on the validation set and return the loss."""
		model.eval()
		eval_loss = 0
		display_interval = 50
		correct_predictions = 0

		with torch.no_grad():

			for idx, (inputs, targets) in enumerate(self.eval_dataloader):
				inputs, targets = inputs.to(device), targets.to(device)
				outputs, _ = model(inputs)
				#outputs = model(inputs)
				loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
				eval_loss += loss.item()


				predictions = outputs.argmax(dim=-1)
				correct_predictions += (predictions == targets).sum().item()
				#total_samples += targets.size(0)

				predicted_tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in predictions[0].tolist()]
				input_tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in inputs[0].cpu().numpy().tolist()]
				target_tokens = [list(self.vocab.keys())[list(self.vocab.values()).index(idx)] for idx in targets[0].cpu().numpy().tolist()]


				# Occasionally print input, target, and prediction for interpretation
				if idx % display_interval == 0:
					print(f"Input: {' '.join(input_tokens)}")
					print(f"Target: {' '.join(target_tokens)}")
					pad_pos = predicted_tokens.index('<PAD>')
					refinedPred = ' '.join(predicted_tokens[0:pad_pos])
					print(f"Prediction: {refinedPred}\n")

		print(f"Correct Predictions: ".format(correct_predictions))
		return eval_loss / len(self.eval_dataloader)

	def tune(self):
		"""Perform hyperparameter tuning by iterating over all parameter combinations."""
		best_params = None
		best_loss = float('inf')
		results = []

		# Iterate over all combinations of hyperparameters
		for param_set in product(*self.param_grid.values()):
			params = dict(zip(self.param_grid.keys(), param_set))
			print(f"\nTraining with parameters: {params}")

			# Initialize model, optimizer, and loss function
			model = self.model_class(
				self.vocab_size,
				params['embedding_dim'],
				params['hidden_dim'],
				params['output_dim'],
				params['n_layers'],
				params['dropout']
			)

			model = model.to(device)
			criterion = nn.CrossEntropyLoss().to(device)  # Move loss function to GPU
			optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

			# Train model
			self.train_model(model, optimizer, criterion, params['epochs'])

			# Evaluate model
			eval_loss = self.evaluate_model(model, criterion)
			results.append((params, eval_loss))

			# Track best parameters
			if eval_loss < best_loss:
				best_loss = eval_loss
				best_params = params

			print(f"Evaluation Loss: {eval_loss:.4f}")

		print(f"\nBest Parameters: {best_params} with Loss: {best_loss:.4f}")
		return best_params, results


# 7. Main Pipeline
def main(hp=1, mode="training"):
	with open(SAMPLE_DATA_FILE) as f:
		methods = [item.rstrip().split() for item in f.readlines()] 
		
	with open(INPUT_TRAIN_PATH) as f:
		input_train_dataset = [item.rstrip().split() for item in f.readlines()]

	with open(TARGET_TRAIN_PATH) as f:
		target_train_dataset = [item.rstrip().split() for item in f.readlines()]

	with open(INPUT_EVAL_PATH) as f:
		input_eval_dataset = [item.rstrip().split() for item in f.readlines()]

	with open(TARGET_EVAL_PATH) as f:
		target_eval_dataset = [item.rstrip().split() for item in f.readlines()]

	with open(INPUT_TEST_PATH) as f:
		input_test_dataset = [item.rstrip().split() for item in f.readlines()]

	with open(TARGET_TEST_PATH) as f:
		target_test_dataset = [item.rstrip().split() for item in f.readlines()]

	#print(methods)

	# Create vocabulary
	vocab = generate_vocab_with_completion_n(methods)
	vocab_size = len(vocab)

	#print(vocab)
	#print(vocab_size)

	# # Create dataset and dataloader
	datasetTrain = CodeCompletionDataset(input_train_dataset, target_train_dataset, vocab, seq_length=100)
	datasetTest = CodeCompletionDataset(input_test_dataset, target_test_dataset, vocab, seq_length=100)
	datasetEval = CodeCompletionDataset(input_eval_dataset, target_eval_dataset, vocab, seq_length=100)

	dataloader_train = DataLoader(datasetTrain, batch_size=32, shuffle=True)
	dataloader_test = DataLoader(datasetTest, batch_size=32, shuffle=True)
	dataloader_eval = DataLoader(datasetEval, batch_size=32, shuffle=True)

	if hp==1:

		######## HP-Tuning Config ########
		# Best Parameters: {'embedding_dim': 256, 'hidden_dim': 256, 'output_dim': 488, 'n_layers': 2, 'learning_rate': 0.001, 'dropout': 0.2, 'epochs': 5} with Loss: 0.8346
		param_grid = {
			'embedding_dim': [128, 256],
			'hidden_dim': [128, 256],
			'output_dim': [vocab_size],
			'n_layers': [2,3],
			'learning_rate': [0.0001, 0.001],
			'dropout': [0.2],
			'epochs': [3,5]
		}

		tuner = HyperparameterTuner(LSTMModel, param_grid, dataloader_train, dataloader_eval, vocab_size, vocab)
		best_params, tuning_results = tuner.tune()

	else:

		#Hyperparameters
		embedding_dim = 64
		hidden_dim = 256
		output_dim = vocab_size
		n_layers = 16
		learning_rate = 0.001
		epochs = 5
		dropout = 0.2


		# Initialize model, optimizer, and loss function
		model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

		if mode == "training":

			optimizer = optim.Adam(model.parameters(), lr=learning_rate)
			criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"]).to(device)

			# Train the model
			train_model(model, dataloader, optimizer, criterion, epochs)

		elif mode == "eval":
			model.load_state_dict(torch.load("models/dl_code_completion.pth"))

			with torch.no_grad():
				for inputs, targets in self.test_dataloader:
					inputs, targets = inputs.to(device), targets.to(device)
					outputs, _ = model(inputs)

					predicted = torch.argmax(outputs, dim=-1)
					#predicted_tokens = [list(vocab.keys())[list(vocab.values()).index(idx)] for idx in predicted[0].tolist()]
					idx_to_token = {idx: token for token, idx in self.vocab.items()}
					predicted_tokens = [idx_to_token.get(idx, "<UNK>") for idx in predictions[0].cpu().tolist()]

					print(predicted_tokens)

		else:
			sys.exit(-1)


if __name__ == "__main__":
	main(hp=1)