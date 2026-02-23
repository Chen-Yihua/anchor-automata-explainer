import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class SimpleBinarySequenceClassifier:
    """
    適用於二元分類（如 regex 任務）的簡單序列分類器 (PyTorch 版本)。
    輸入: 字元序列 (list of list of str)
    輸出: 0/1
    """
    class _LSTMClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, lstm_units, max_len):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)
            self.dropout = Dropout(0.5)
            self.fc = nn.Linear(lstm_units * 2, 1)
            self.max_len = max_len
        def forward(self, x):
            x = self.embedding(x)
            _, (h_n, _) = self.lstm(x)
            h_cat = torch.cat([h_n[-2], h_n[-1]], dim=1)  # concat forward/backward
            out = self.dropout(h_cat)
            out = self.fc(out)
            return torch.sigmoid(out)

    def __init__(self, max_len=32, embedding_dim=64, lstm_units=64, device=None):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.char2idx = None
        self.idx2char = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, epochs=30, batch_size=64, verbose=0, early_stopping_rounds=5, val_split=0.1):
        # 建立字元表
        chars = sorted(set(c for seq in X for c in seq))
        self.char2idx = {c: i+1 for i, c in enumerate(chars)}  # 0 for padding
        self.idx2char = {i+1: c for i, c in enumerate(chars)}
        vocab_size = len(self.char2idx) + 1
        # 轉為 index
        X_idx = [[self.char2idx.get(c, 0) for c in seq] for seq in X]
        X_pad = np.array([seq + [0]*(self.max_len-len(seq)) if len(seq)<self.max_len else seq[:self.max_len] for seq in X_idx])
        y = np.array(y).astype(np.float32)
        # PyTorch tensors
        X_tensor = torch.LongTensor(X_pad)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        # validation split
        val_size = int(len(X_tensor) * val_split)
        if val_size > 0:
            X_val, y_val = X_tensor[:val_size], y_tensor[:val_size]
            X_train, y_train = X_tensor[val_size:], y_tensor[val_size:]
        else:
            X_train, y_train = X_tensor, y_tensor
            X_val, y_val = None, None
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # 建立模型
        self.model = self._LSTMClassifier(vocab_size, self.embedding_dim, self.lstm_units, self.max_len).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        best_val_loss = float('inf')
        no_improve = 0
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                output = self.model(xb)
                loss = criterion(output, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss/len(train_dataset)
            val_loss = None
            if val_size > 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val.to(self.device))
                    val_loss = criterion(val_pred, y_val.to(self.device)).item()
                self.model.train()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
                if no_improve >= early_stopping_rounds:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, X):
        self.model.eval()
        X_idx = [[self.char2idx.get(c, 0) for c in seq] for seq in X]
        X_pad = np.array([seq + [0]*(self.max_len-len(seq)) if len(seq)<self.max_len else seq[:self.max_len] for seq in X_idx])
        X_tensor = torch.LongTensor(X_pad).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        return (y_pred > 0.5).astype(int)

    def predict_proba(self, X):
        self.model.eval()
        X_idx = [[self.char2idx.get(c, 0) for c in seq] for seq in X]
        X_pad = np.array([seq + [0]*(self.max_len-len(seq)) if len(seq)<self.max_len else seq[:self.max_len] for seq in X_idx])
        X_tensor = torch.LongTensor(X_pad).to(self.device)
        with torch.no_grad():
            y_pred = self.model(X_tensor).cpu().numpy().flatten()
        return np.stack([1-y_pred, y_pred], axis=1)
