
import sys, os
import torch
import random
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

# Put modified_modules at the front so 'alibi' resolves to local version
for path in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from datasets.tomita_loader_dfa import (
    L1, L2, L2A, L3, L3A, L4, L4A, L5, L5A, L6, L6A, L7, L7A, L8A, L9A, L10A,
    L2AB, L3AB, L4AB, L5AB,
)
from tee import Tee
import numpy as np
import time
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelBinarizer

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Language Models to Benchmark ---
LANGUAGES = {
    # "L1 (Constant Sequence)": L1,
    # "L2 (Pattern (abcd)*)": L2,
    # "L2A (Pattern (aa)*)": L2A,``
    # "L2AB (Pattern (abcdabcd)*)": L2AB,
    # "L3 (Parity of counts)": L3,
    # "L3A (Pattern (aaa)*)": L3A,
    "L3AB (Pattern (abcdabcdabcd)*)": L3AB, # 30 states: ['a', 'b', 'c', 'd'] * 6 -
    "L4 (No three consecutive same)": L4,
    # "L4A (Pattern (aaaa)*)": L4A,
    # "L4AB (Pattern (abcdabcdabcdabcd)*)": L4AB,
    "L5 (Even count of A and B)": L5,
    # "L5A (Pattern (aaaaa)*)": L5A,
    # "L5AB (Pattern (abcdabcdabcdabcdabcd)*)": L5AB,
    "L6 (Modulo Balance)": L6,
    # "L6A (Pattern (aaaaaa)*)": L6A,
    "L7 (Pattern a*b*a*b*)": L7,
    # "L7A (Pattern (aaaaaaa)*)": L7A,
    # "L8A (Pattern (aaaaaaaa)*)": L8A,
    # "L9A (Pattern (aaaaaaaaa)*)": L9A,
    # "L10A (Pattern (aaaaaaaaaa)*)": L10A,

    # "EvenPairs (Even ab/ba pairs)": EvenPairs,
}

# --- Main Loop ---
for lang_name, lang_class in LANGUAGES.items():
    lang_code = lang_name.split(" ")[0]
    
    # Create output directory for this language
    output_dir = os.path.join(PROJECT_ROOT, "test_result", f"tomita_{lang_code}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f"tomita_{lang_code}.txt")
    with open(output_filename, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)

        print(f"\n{'='*80}")
        print(f"Running tomita for: {lang_name}")
        print(f"{'='*80}\n")
        
        # Start timing
        total_start_time = time.time()

        # Generate Data
        data_gen_start = time.time()
        model = lang_class()
        pos_samples, neg_samples = model.generate_samples(num_pos=5000, num_neg=5000, max_length=9)
        data_gen_time = time.time() - data_gen_start
        print(f"Data generation time: {data_gen_time:.2f}s")

        X = pos_samples + neg_samples
        y = [1] * len(pos_samples) + [0] * len(neg_samples)
        X = np.array(X, dtype=object)
        y = np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # PyTorch LSTM for variable-length sequence classification
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import Dataset, DataLoader
        from sklearn.metrics import accuracy_score

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        alphabet = ['a', 'b', 'c', 'd']
        char2idx = {c: i+1 for i, c in enumerate(alphabet)}  # 0 for padding
        pad_idx = 0

        def encode_seq(seq):
            def to_char(val):
                if isinstance(val, (float, np.floating, int, np.integer)):
                    if int(val) == 0:
                        return 'a'
                    elif int(val) == 1:
                        return 'b'
                    else:
                        raise ValueError(f'Unknown value: {val}')
                return val
            return [char2idx[to_char(c)] for c in seq]

        def pad_sequences(seqs):
            # 若序列長度為0，補一個pad_idx
            seqs = [s if len(s) > 0 else [pad_idx] for s in seqs]
            maxlen = max(len(s) for s in seqs)
            return [s + [pad_idx]*(maxlen-len(s)) for s in seqs], [len(s) for s in seqs]

        class SeqDataset(Dataset):
            def __init__(self, X, y):
                self.X = [encode_seq(s) for s in X]
                self.y = y
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)

        from models.simple_sequence_classifier import SimpleSequenceClassifier

        train_dataset = SeqDataset(X_train, y_train)
        test_dataset = SeqDataset(X_test, y_test)
        def collate_fn(batch):
            xs = [b[0] for b in batch]
            ys = [b[1] for b in batch]
            lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
            xs_padded = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_idx)
            ys = torch.stack(ys)
            return xs_padded, ys, lengths
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        model_rnn = SimpleSequenceClassifier(vocab_size=len(alphabet)+1, emb_dim=16, hidden_dim=64).to(device)
        optimizer = optim.Adam(model_rnn.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(20):
            model_rnn.train()
            epoch_loss = 0.0
            batch_count = 0
            for xb, yb, lb in train_loader:
                xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
                # 過濾掉長度為 0 的序列
                mask = lb > 0
                if not torch.all(mask):
                    xb = xb[mask]
                    yb = yb[mask]
                    lb = lb[mask]
                if xb.size(0) == 0:
                    continue  # 跳過這個 batch
                optimizer.zero_grad()
                out = model_rnn(xb, lb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
            if batch_count > 0:
                avg_loss = epoch_loss / batch_count
                print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        def predict_rnn(X):
            model_rnn.eval()
            X_enc = [encode_seq(s) for s in X]
            X_pad, lengths = pad_sequences(X_enc)
            xb = torch.tensor(X_pad, dtype=torch.long).to(device)
            lb = torch.tensor(lengths, dtype=torch.long).to(device)
            with torch.no_grad():
                logits = model_rnn(xb, lb)
                probs = torch.sigmoid(logits)
                return (probs.cpu().numpy() > 0.5).astype(int)

        train_acc = accuracy_score(y_train, predict_rnn(X_train))
        test_acc = accuracy_score(y_test, predict_rnn(X_test))
        print(f"Predictor Train Accuracy: {train_acc:.3f}")
        print(f"Predictor Test Accuracy: {test_acc:.3f}")
        # print("[DEBUG] predictiom of ['a', 'a']:", predict_rnn([['a', 'a']]))
        # print("[DEBUG] predictiom of ['a', 'a', 'a']:", predict_rnn([['a', 'a', 'a']]))
        
        test_instance = ['a', 'b', 'c', 'd'] * 6  # Example test instance
        # print(f"Alphabet: {alphabet}")
        print(f"Test instance: {test_instance}")

        explainer = AnchorTabular(
            predictor=predict_rnn,
            feature_names=[f'seq_{i}' for i in range(max(len(s) for s in X_train))],
            categorical_names={},
            seed=1
        )
        explainer.fit(
            automaton_type='DFA',
            train_data=X_train,
            alphabet=alphabet,
            disc_perc=None
        )
        explainer.samplers[0].d_train_data = X_train
        
        # run explainer.explain (beam search)
        accuracy_threshold = 0.95
        state_threshold = 5
        delta = 0.01
        tau = 0.01
        batch_size = 500
        coverage_samples = 1000
        beam_size = 3
        # max_anchor_size = None
        min_samples_start = 3000
        # n_covered_ex = 20
        edit_distance = 24
        
        # Start timing for DFA learning
        dfa_start_time = time.time()
        explanation = explainer.explain(
            type='Tabular',
            automaton_type='DFA',
            alphabet=alphabet,
            X=test_instance,
            edit_distance=edit_distance,
            accuracy_threshold=accuracy_threshold,
            state_threshold=state_threshold,
            delta=delta,
            tau=tau,
            beam_size=beam_size,
            batch_size=batch_size,
            min_samples_start=min_samples_start,
            # n_covered_ex=n_covered_ex,
            verbose=True, # Set to True for detailed beam search logs
            constants=[],
            output_dir=output_dir,
        )
        dfa_learning_time = time.time() - dfa_start_time
        total_time = time.time() - total_start_time

        # 5. Print Results
        print('\n============== Final Result ==============')
        print(f"Explaining instance: {test_instance}")
        print(f"Prediction: {predict_rnn([test_instance])}")
        
        if explanation and explanation.data:
            dfa = explanation.data.get('automata', None)
            print('DFA:', dfa)
            print('Training Accuracy (擾動樣本):', explanation.data.get('training_accuracy', 'N/A'))
            print('Testing Accuracy (擾動樣本):', explanation.data.get('testing_accuracy', 'N/A'))
            print('Number of States:', explanation.data.get('state', 'N/A'))
        else:
            print("Explanation failed or did not produce a valid automaton.")
        
        # Print timing information
        print(f"\n============== Timing ==============")
        print(f"Data generation time: {data_gen_time:.2f}s")
        print(f"DFA learning time: {dfa_learning_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Results for {lang_code} saved to: {output_dir}")

print(f"\n{'='*60}")
print(f"All benchmark results saved to: test_result/benchmark_*/")
print(f"{'='*60}")
