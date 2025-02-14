import torch
import torch.nn as nn
import torch.optim as optim
import random
import string

# Set random seed for reproducibility
torch.manual_seed(99)  # 随机种子


# Define the RNN model
class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)  # 代表输入的第一个维度是一个batch
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, _ = self.rnn(embedded)
        final_hidden_state = rnn_out[:, -1, :]  # Get the last hidden state
        output = self.fc(final_hidden_state)
        return output


# Generate random strings and labels
def generate_data(num_samples=1000, max_length=20):
    data = []
    labels = []  # 真实类别
    all_chars = string.ascii_lowercase  # All lowercase English letters

    for _ in range(num_samples):
        length = random.randint(1, max_length)
        string_input = ''.join(random.choices(all_chars, k=length))
        label = string_input.find('x')  # Find the first appearance of 'x'
        if label == -1:  # If no 'x' is found, use the max length as label
            label = max_length
        data.append(string_input)
        labels.append(label)

    return data, labels


# Convert string to a tensor of indices
def string_to_tensor(s, vocab):
    return torch.tensor([vocab[char] for char in s], dtype=torch.long)


# Create vocabulary mapping lowercase letters to indices
vocab = {ch: idx for idx, ch in enumerate(string.ascii_lowercase)}
vocab_size = len(vocab)  # 26 characters
max_length = 20  # Max length of the strings

# Generate the training data
data, labels = generate_data(num_samples=1000)

# Convert the strings and labels to tensors
inputs = [string_to_tensor(s, vocab) for s in data]
inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=vocab['a'])  # Padding with 'a' index
labels = torch.tensor(labels, dtype=torch.long)

# Split into training and test sets
train_size = int(0.8 * len(data))
train_inputs, test_inputs = inputs[:train_size], inputs[train_size:]
train_labels, test_labels = labels[:train_size], labels[train_size:]

# Initialize the model
hidden_size = 128
output_size = max_length + 1  # Class labels from 0 to max_length
model = RNNClassifier(input_size=vocab_size, hidden_size=hidden_size, output_size=output_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()  # criterion = loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(train_inputs.size(0))  # 把训练数据再打乱
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for i in range(0, train_inputs.size(0), batch_size):
        indices = permutation[i:i + batch_size]  # 下标
        batch_inputs = train_inputs[indices]
        batch_labels = train_labels[indices]

        optimizer.zero_grad()  # 每一个batch结束之后就清零

        # Forward pass
        outputs = model(batch_inputs)

        # Compute loss
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()  # 优化器优化一次

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == batch_labels).sum().item()
        total_predictions += batch_labels.size(0)

        running_loss += loss.item()

    # Print stats for each epoch
    epoch_loss = running_loss / (len(train_inputs) // batch_size)
    accuracy = correct_predictions / total_predictions
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

# Evaluate on the test set
model.eval()  # 代表进入评估模式，该模式下所有的计算都不会累计梯度
test_outputs = model(test_inputs)
_, test_predictions = torch.max(test_outputs, 1)
test_accuracy = (test_predictions == test_labels).sum().item() / test_labels.size(0)
print(f'Test Accuracy: {test_accuracy:.4f}')

