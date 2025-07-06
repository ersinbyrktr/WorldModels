# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there"))
print(decode(encode("hii there")))


class TinyShakespeare(Dataset):
    def __init__(self, text, block_size):
        self.data = torch.tensor(encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1: idx + 1 + self.block_size]
        return x, y


n = int(0.9 * len(text))
train_text = text[:n]
val_text = text[n:]

tiny_train = TinyShakespeare(train_text, block_size=512)
tiny_val = TinyShakespeare(val_text, block_size=512)

train_loader = DataLoader(tiny_train, batch_size=512, shuffle=True)
test_loader = DataLoader(tiny_val, batch_size=512, shuffle=False)


# @title RNN
class RNN(nn.Module):
    def __init__(self, d_embed, d_hidden, vocab_size):
        super().__init__()
        self.d_hidden = d_hidden

        # project input embedding → hidden
        self.input_projector = nn.Linear(d_embed, d_hidden)
        # project previous hidden → hidden
        self.hidden_projector = nn.Linear(d_hidden, d_hidden)
        # project hidden → logits over vocab
        self.out_projector = nn.Linear(d_hidden, vocab_size)

        # token embedding table
        self.embedding_table = nn.Embedding(vocab_size, d_embed)

    def forward(self, idx, hidden):
        """
        idx    : (B,) LongTensor of token indices for *one* timestep
        hidden : (B, d_hidden) previous hidden state
        """
        # lookup embedding for this timestep
        x = self.embedding_table(idx)  # → (B, d_embed)
        # compute next hidden
        hidden_next = self.input_projector(x) \
                      + self.hidden_projector(hidden)
        hidden_next = F.relu(hidden_next)  # nonlinearity
        # project to vocab logits
        logits = self.out_projector(hidden_next)  # → (B, vocab_size)
        return logits, hidden_next

    def generate(self, idx, max_new_tokens):
        """
        idx            : (B, T) LongTensor of seed token indices
        max_new_tokens : how many new tokens to sample
        Returns       : (B, T + max_new_tokens) LongTensor
        """
        B, T = idx.shape
        device = idx.device

        # start with zero hidden state on the right device
        hidden = torch.zeros(B, self.d_hidden, device=device)

        # “prime” the RNN by consuming all seed tokens
        for t in range(T):
            logits, hidden = self(idx[:, t], hidden)

        output_indices = [idx]  # list of (B, current_length) tensors

        # sample first next token outside the loop for clarity
        probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
        idx_next = torch.multinomial(probs, num_samples=1).squeeze(1)  # → (B,)
        output_indices.append(idx_next.unsqueeze(1))  # make it (B,1)

        # now generate further tokens
        for _ in range(max_new_tokens - 1):  # already did one
            logits, hidden = self(idx_next, hidden)  # step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).squeeze(1)
            output_indices.append(idx_next.unsqueeze(1))

        # concatenate along time dimension
        return torch.cat(output_indices, dim=1)  # (B, T + max_new_tokens)

    def train_model(self, train_loader, test_loader, n_epochs, lr):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for epoch in range(n_epochs):

            self.train()
            total_train_loss = 0
            n_sample = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")

            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                B, T = x.shape
                hidden = torch.zeros(x.shape[0], self.d_hidden, device=x.device)

                logits_seq = []
                for t in range(T):
                    logits, hidden = self(x[:, t], hidden)
                    logits_seq.append(logits)

                logits = torch.stack(logits_seq, dim=1)  # (B, T, vocab_size)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * B
                n_sample += B

                running_training_loss = total_train_loss / n_sample
                pbar.set_postfix({"Running Training Loss": f"{running_training_loss:.4f}"})

            with torch.no_grad():
                self.eval()
                total_val_loss = 0
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    B, T = x.shape
                    hidden = torch.zeros(x.shape[0], self.d_hidden, device=x.device)

                    logits_seq = []
                    for t in range(T):
                        logits, hidden = self(x[:, t], hidden)
                        logits_seq.append(logits)

                    logits = torch.stack(logits_seq, dim=1)  # (B, T, vocab_size)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="mean")

                    total_val_loss += loss.item() * B

                data = torch.tensor(encode("\n"), dtype=torch.long, device=device).reshape(1, -1)
                response = self.generate(data, max_new_tokens=100)
                t = decode(response[0].tolist())
                print(f"Some random text:{t}")

            print(
                f"Epoch: {epoch + 1} Train Loss: {(total_train_loss / len(train_loader.dataset))} Val Loss: {(total_val_loss / len(test_loader.dataset))}")


# @title Deep RNN
class RNN2(nn.Module):
    def __init__(self, d_embed, d_hidden, vocab_size):
        super().__init__()
        self.d_hidden = d_hidden

        self.input_projector_1 = nn.Linear(d_embed, d_hidden)
        self.hidden_projector_1 = nn.Linear(d_hidden, d_hidden)
        self.ln1 = nn.LayerNorm(d_hidden)

        self.ln_between = nn.LayerNorm(d_hidden)

        self.out_projector_1 = nn.Linear(d_hidden, d_hidden)

        self.input_projector_2 = nn.Linear(d_hidden, d_hidden)
        self.hidden_projector_2 = nn.Linear(d_hidden, d_hidden)
        self.out_projector_2 = nn.Linear(d_hidden, vocab_size)
        self.ln2 = nn.LayerNorm(d_hidden)

        # token embedding table
        self.embedding_table = nn.Embedding(vocab_size, d_embed)

    def forward(self, idx, hidden_1, hidden_2):
        """
        idx    : (B,) LongTensor of token indices for *one* timestep
        hidden : (B, d_hidden) previous hidden state
        """
        # lookup embedding for this timestep
        x = self.embedding_table(idx)  # → (B, d_embed)
        # compute next hidden

        hidden_next_1 = F.relu(self.ln1(self.input_projector_1(x) + self.hidden_projector_1(hidden_1)))

        out_1 = self.ln_between(self.out_projector_1(hidden_next_1))

        hidden_next_2 = F.relu(self.ln2(self.input_projector_2(out_1) + self.hidden_projector_2(hidden_2)))

        logits = self.out_projector_2(hidden_next_2)

        return logits, hidden_next_1, hidden_next_2

    def generate(self, idx, max_new_tokens):
        """
        idx            : (B, T) LongTensor of seed token indices
        max_new_tokens : how many new tokens to sample
        Returns       : (B, T + max_new_tokens) LongTensor
        """
        B, T = idx.shape
        device = idx.device

        # start with zero hidden state on the right device
        hidden_1 = torch.zeros(B, self.d_hidden, device=device)
        hidden_2 = torch.zeros(B, self.d_hidden, device=device)

        # “prime” the RNN by consuming all seed tokens
        for t in range(T):
            logits, hidden_1, hidden_2 = self(idx[:, t], hidden_1, hidden_2)

        output_indices = [idx]  # list of (B, current_length) tensors

        # sample first next token outside the loop for clarity
        probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
        idx_next = torch.multinomial(probs, num_samples=1).squeeze(1)  # → (B,)
        output_indices.append(idx_next.unsqueeze(1))  # make it (B,1)

        # now generate further tokens
        for _ in range(max_new_tokens - 1):  # already did one
            logits, hidden_1, hidden_2 = self(idx_next, hidden_1, hidden_2)  # step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1).squeeze(1)
            output_indices.append(idx_next.unsqueeze(1))

        # concatenate along time dimension
        return torch.cat(output_indices, dim=1)  # (B, T + max_new_tokens)

    def train_model(self, train_loader, test_loader, n_epochs, lr):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        for epoch in range(n_epochs):

            self.train()
            total_train_loss = 0
            n_sample = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}")

            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                B, T = x.shape
                hidden_1 = torch.zeros(x.shape[0], self.d_hidden, device=x.device)
                hidden_2 = torch.zeros(x.shape[0], self.d_hidden, device=x.device)

                logits_seq = []
                for t in range(T):
                    logits, hidden_1, hidden_2 = self(x[:, t], hidden_1, hidden_2)
                    logits_seq.append(logits)

                logits = torch.stack(logits_seq, dim=1)  # (B, T, vocab_size)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="mean")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * B
                n_sample += B

                running_training_loss = total_train_loss / n_sample
                pbar.set_postfix({"Running Training Loss": f"{running_training_loss:.4f}"})

            with torch.no_grad():
                self.eval()
                total_val_loss = 0
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    B, T = x.shape
                    hidden_1 = torch.zeros(x.shape[0], self.d_hidden, device=x.device)
                    hidden_2 = torch.zeros(x.shape[0], self.d_hidden, device=x.device)

                    logits_seq = []
                    for t in range(T):
                        logits, hidden_1, hidden_2 = self(x[:, t], hidden_1, hidden_2)
                        logits_seq.append(logits)

                    logits = torch.stack(logits_seq, dim=1)  # (B, T, vocab_size)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="mean")

                    total_val_loss += loss.item() * B

                data = torch.tensor(encode("\n"), dtype=torch.long, device=device).reshape(1, -1)
                response = self.generate(data, max_new_tokens=100)
                t = decode(response[0].tolist())
                print(f"Some random text:{t}")

            print(
                f"Epoch: {epoch + 1} Train Loss: {(total_train_loss / len(train_loader.dataset))} Val Loss: {(total_val_loss / len(test_loader.dataset))}")


# rnn = RNN(d_embed=128, d_hidden=256, vocab_size=vocab_size)

rnn2 = RNN2(d_embed=128, d_hidden=512, vocab_size=vocab_size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
data = torch.tensor(encode("\n"), dtype=torch.long, device=device).reshape(1, -1)

# response = rnn.generate(data, max_new_tokens=100)
# print(decode(response[0].tolist()))

# response = rnn.generate(data, max_new_tokens=300)
# print(decode(response[0].tolist()))

# response = rnn.generate(data, max_new_tokens=300)
# print(decode(response[0].tolist()))

# rnn.train_model(train_loader, test_loader, n_epochs=1, lr=1e-5)

rnn2.train_model(train_loader, test_loader, n_epochs=1, lr=1e-5)

idx = encode("\n")
t = torch.tensor(idx, dtype=torch.long, device=device).reshape(1, -1)
response = rnn2.generate(t, max_new_tokens=300)
print(decode(response[0].tolist()))
