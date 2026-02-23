import torch
import torch.nn as nn

class SimpleSequenceClassifier(nn.Module):
    """
    A simple LSTM-based sequence classifier for variable-length categorical sequences.
    Usage:
        model = SimpleSequenceClassifier(vocab_size=..., emb_dim=..., hidden_dim=...)
        logits = model(x, lengths)  # x: (batch, seq_len), lengths: (batch,)
    """
    def __init__(self, vocab_size, emb_dim=16, hidden_dim=32):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        # 過濾掉長度為 0 的序列
        mask = lengths > 0
        if not torch.all(mask):
            x = x[mask]
            lengths = lengths[mask]
        if x.size(0) == 0:
            # 若全為空，回傳空 tensor
            return torch.empty(0, device=x.device)
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        out = self.fc(h[-1])
        return out.squeeze(-1)
