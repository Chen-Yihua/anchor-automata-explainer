# models/sequence_classifier.py


# RNN-based sequence classifier using PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os

class SimpleSequenceClassifier:
    def __init__(self, max_len=50, embedding_dim=16, rnn_units=32, num_layers=2, dropout=0.3, device=None):
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.symbol2id = None
        self.id2symbol = None
        self.num_classes = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y, epochs=10, batch_size=32):
        X_pad, vocab_size = self._prepare_X(X, fit_vocab=True)
        self.vocab_size = vocab_size
        y = np.array(y)
        num_classes = len(np.unique(y))
        self.num_classes = num_classes
        self.model = _TorchRNNClassifier(
            self.vocab_size+1,
            self.embedding_dim,
            self.rnn_units,
            self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        X_tensor = torch.LongTensor(X_pad)
        y_tensor = torch.LongTensor(y)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        steps_per_epoch = len(loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.003, epochs=epochs,
            steps_per_epoch=steps_per_epoch, pct_start=0.3,
            anneal_strategy='cos'
        )
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for i, (xb, yb) in enumerate(loader):
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(xb)
                loss = criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item() * xb.size(0)
                total_correct += (torch.argmax(out, dim=1) == yb).sum().item()
                total_samples += xb.size(0)
            avg_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, LR: {lr:.6f}")

    def save(self, model_path):
        """Save model and vocabulary to file."""
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        checkpoint = {
            'model_state': self.model.state_dict(),
            'symbol2id': self.symbol2id,
            'id2symbol': self.id2symbol,
            'num_classes': self.num_classes,
            'vocab_size': self.vocab_size,
            'max_len': self.max_len,
            'embedding_dim': self.embedding_dim,
            'rnn_units': self.rnn_units,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path):
        """Load model and vocabulary from file."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.symbol2id = checkpoint['symbol2id']
        self.id2symbol = checkpoint['id2symbol']
        self.num_classes = checkpoint['num_classes']
        self.vocab_size = checkpoint['vocab_size']
        self.max_len = checkpoint['max_len']
        self.embedding_dim = checkpoint['embedding_dim']
        self.rnn_units = checkpoint['rnn_units']
        self.num_layers = checkpoint['num_layers']
        self.dropout = checkpoint['dropout']
        
        self.model = _TorchRNNClassifier(
            self.vocab_size + 1,
            self.embedding_dim,
            self.rnn_units,
            self.num_classes,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def predict(self, X, batch_size=128):
        X_pad, _ = self._prepare_X(X, fit_vocab=False)
        X_tensor = torch.LongTensor(X_pad).to(self.device)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                logits = self.model(batch)
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(batch_preds)
        return np.concatenate(preds)

    def _prepare_X(self, X, fit_vocab=False):
        # Flatten nested sequences and collect all symbols
        seqs = []
        all_symbols = set()
        for seq in X:
            if isinstance(seq, (list, np.ndarray)) and len(seq) > 0:
                if isinstance(seq[0], (list, np.ndarray)):
                    seq = seq[0]
            for s in seq:
                all_symbols.add(s)
            seqs.append(seq)
        # Build vocab if needed
        if fit_vocab or self.symbol2id is None:
            symbol2id = {s: i+1 for i, s in enumerate(sorted(all_symbols, key=str))}  # 0 reserved for padding
            id2symbol = {i: s for s, i in symbol2id.items()}
            self.symbol2id = symbol2id
            self.id2symbol = id2symbol
        # Encode
        seqs_id = [[self.symbol2id.get(s, 0) for s in seq] for seq in seqs]
        # Pad
        X_pad = np.zeros((len(seqs_id), self.max_len), dtype=np.int64)
        for i, seq in enumerate(seqs_id):
            l = min(len(seq), self.max_len)
            X_pad[i, :l] = seq[:l]
        vocab_size = max(self.symbol2id.values()) if self.symbol2id else 1
        return X_pad, vocab_size


class _TorchRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, rnn_units, num_classes, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            rnn_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.bn = nn.BatchNorm1d(rnn_units * 2)
        self.dense = nn.Linear(rnn_units * 2, rnn_units * 2)
        self.fc = nn.Linear(rnn_units * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        # 雙向 LSTM，h_n shape: (num_layers * 2, batch, rnn_units)
        # 取最後一層的正向和反向 hidden，然後拼接
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h = torch.cat([h_forward, h_backward], dim=1)  # (batch, rnn_units*2)
        h = self.bn(h)
        h = torch.relu(self.dense(h))
        out = self.fc(h)
        return out
