# GenAI for Software Development: Extra Credit Assignment

---

# **1. Introduction**
This project focuses on code completion in Java using a neural network-based approach. It builds on the traditional N-gram language modeling technique by incorporating modern deep learning methods, such as Recurrent Neural Networks (RNNs), to predict the next token in a Java code sequence. The model learns token patterns and probabilities from training data, enabling it to generate accurate predictions for masked tokens. This approach has applications in software engineering automation and natural language processing.

The repository includes all necessary scripts for preprocessing, training, and evaluating the model. The dataset used for training is preprocessed Java code, and the implementation leverages PyTorch for model development.

---

# **2. Getting Started**
This project requires Python 3.9+ and is compatible with macOS, Linux, and Windows.

## **2.1 Preparations**
Clone the repository to your local workspace:
```
git clone https://github.com/jdkuffa/cs420-extracredit.git
```

Navigate into the repository:

```
cd cs420-extracredit
```

Set up a virtual environment and activate it:

### For macOS/Linux:

```
python -m venv ./venv/
```
```
source venv/bin/activate
```

### For Windows:

```
python -m venv ./venv/
```
```
.\venv\Scripts\activate
```

To deactivate the virtual environment, use the command:

```
deactivate
```

## **2.2 Install Packages**
Install all required dependencies listed in requirements.txt:

```
pip install -r requirements.txt
```
## 2.3 Run Code Completion Model
**Prepare Dataset:** Ensure your dataset file is located at data/sample-pt.csv. You can replace this path with your own dataset file if needed.
**Run Training and Evaluation:** Execute the main script to train and evaluate the RNN-based model on the provided dataset:

```
python extracredit.py
```

Output:

The training performance log will be saved as ```results/training_performance.csv```.

A plot of training loss over epochs will be displayed.

---

# 3. **Key Features of the Codebase**
Data Preprocessing:
- Cleaning: Removes invalid characters from Java code.
- Tokenization: Converts Java code into a sequence of tokens using javalang.
- Masking: Randomly masks 15% of tokens for prediction during training.

Model Architecture:
- Embedding Layer: Converts tokens into dense vector representations.
- RNN (LSTM): Processes sequential data to capture token dependencies.
- Fully Connected Layer: Outputs probabilities for each token in the vocabulary.

Training:
- Implements cross-entropy loss for optimization.
- Logs performance metrics across epochs.
- Supports batch processing with PyTorch's DataLoader.

---

# 4. Example Workflow
Input Example:
```java
public int add(int a, int b) {
    return a + b;
}
```

Tokenized Output:
```text
['public', 'int', 'add', '(', 'int', 'a', ',', 'int', 'b', ')', '{', 'return', 'a', '+', 'b', ';', '}']
```

Masked Input:
```text
['public', 'int', 'MASK_1', '(', 'int', 'a', ',', 'int', 'b', ')', '{', 'return', 'a', '+', 'b', ';', '}']
```

Model Prediction:
```text
'MASK_1' → 'add'
```

# 5. Additional Information
This project is licensed under the MIT License.
