import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Sample text data
text = "hello world. this is a basic text generation example. How are you? I'm fine, thank you. Let's generate some text using a simple LSTM model. This is a demonstration of how to use PyTorch for text generation tasks. The model will learn to predict the next character in a sequence based on the previous characters."

# Preprocess: Create character mappings
chars = sorted(list(set(text)))
char2idx = {ch: idx for idx, ch in enumerate(chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}

# Hyperparameters
seq_length = 10
hidden_size = 128
batch_size = 16
epochs = 100
lr = 0.003

# Dataset preparation
class TextDataset(Dataset):
    def __init__(self, text, seq_length):
        self.data = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        input_seq = [char2idx[c] for c in chunk[:-1]]
        target_seq = [char2idx[c] for c in chunk[1:]]
        return torch.tensor(input_seq), torch.tensor(target_seq)

dataset = TextDataset(text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model definition
class TextGenModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Initialize model
model = TextGenModel(vocab_size=len(chars), hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs, _ = model(x_batch)
        loss = criterion(outputs.view(-1, len(chars)), y_batch.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Generation function
def generate_text(start_seq="hello", gen_len=50):
    model.eval()
    input_seq = torch.tensor([char2idx[c] for c in start_seq]).unsqueeze(0)
    hidden = None
    generated = start_seq

    for _ in range(gen_len):
        output, hidden = model(input_seq, hidden)
        last_char_logits = output[:, -1, :]
        predicted_id = torch.argmax(last_char_logits, dim=-1).item()
        generated += idx2char[predicted_id]
        input_seq = torch.tensor([[predicted_id]])
    return generated

# Test generation
print(generate_text("how are you", 100))