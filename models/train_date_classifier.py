"""
Train DFA date string classifier with variable-length examples.
Saves the trained model and train/test split for later use in run_date.py.
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

# Put modified_modules at the front so 'alibi' resolves to local version
for path in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from models.sequence_classifier import SequenceClassifier
from sklearn.metrics import accuracy_score
import datetime
import random as _random
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Data generation functions (copied from run_date.py)
def generate_real_dates(n=10000):
    dates = set()
    start = datetime.date(1900, 1, 1)
    end = datetime.date(2026, 12, 31)
    delta_days = (end - start).days
    while len(dates) < n:
        rand_day = start + datetime.timedelta(days=_random.randint(0, delta_days))
        # Generate dates with different separators: - and /
        separators = ['-', '/']
        separator = _random.choice(separators)
        dates.add(rand_day.strftime(f"%m{separator}%d{separator}%Y"))
    return list(dates)

def generate_date_negatives(positive_samples, n=10000):
    alphabet = sorted(set(c for s in positive_samples for c in s))
    negatives = set()
    # Support both - and / and # as separators
    fmt_pattern_dash = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    fmt_pattern_slash = re.compile(r"^\d{2}/\d{2}/\d{4}$")
    fmt_pattern_hash = re.compile(r"^\d{2}#\d{2}#\d{4}$")  # Invalid format with #
    
    def is_valid_date(s):
        if fmt_pattern_dash.fullmatch(s):
            try:
                datetime.datetime.strptime(s, "%m-%d-%Y")
                return True
            except ValueError:
                return False
        elif fmt_pattern_slash.fullmatch(s):
            try:
                datetime.datetime.strptime(s, "%m/%d/%Y")
                return True
            except ValueError:
                return False
        elif fmt_pattern_hash.fullmatch(s):
            # # separator is always invalid
            return False
        return False
    
    attempts = 0
    while len(negatives) < n and attempts < n * 20:
        base = _random.choice(positive_samples)
        char_list = list(base)
        mutation = _random.choice(['delete', 'insert', 'replace', 'swap', 'change_sep'])
        idx = _random.randint(0, len(char_list) - 1)
        
        if mutation == 'delete' and len(char_list) > 1:
            del char_list[idx]
        elif mutation == 'insert':
            char_list.insert(idx, _random.choice(alphabet))
        elif mutation == 'replace':
            original = char_list[idx]
            candidates = [c for c in alphabet if c != original]
            if candidates:
                char_list[idx] = _random.choice(candidates)
        elif mutation == 'swap' and len(char_list) > 1:
            j = _random.randint(0, len(char_list) - 1)
            char_list[idx], char_list[j] = char_list[j], char_list[idx]
        elif mutation == 'change_sep':
            # Replace - or / with #
            mutated = ''.join(char_list)
            if '-' in mutated:
                mutated = mutated.replace('-', '#')
            elif '/' in mutated:
                mutated = mutated.replace('/', '#')
            if not is_valid_date(mutated):
                negatives.add(mutated)
            attempts += 1
            continue
        
        mutated = ''.join(char_list)
        if not is_valid_date(mutated):
            negatives.add(mutated)
        attempts += 1
    
    # Generate additional negatives with # separator
    while len(negatives) < n:
        choice = _random.random()
        if choice < 0.6:
            # Generate invalid dates with # separator
            month = _random.randint(1, 12)
            day = _random.randint(1, 31)
            year = _random.randint(1900, 2026)
            rand_str = f"{month:02d}#{day:02d}#{year}"
            negatives.add(rand_str)
        else:
            # Generate random strings
            length = _random.randint(8, 12)
            rand_str = ''.join(_random.choice(alphabet) for _ in range(length))
            if not is_valid_date(rand_str):
                negatives.add(rand_str)
    
    return list(negatives)

def generate_variable_length_samples(X, y, random_state=42):
    np.random.seed(random_state)
    X_var = []
    y_var = []
    for seq, label in zip(X, y):
        seq_len = len(seq)
        for ratio in [1.0, 0.7, 0.5]:
            new_len = max(1, int(seq_len * ratio))
            truncated_seq = seq[:new_len]
            X_var.append(truncated_seq)
            y_var.append(label)
    indices = np.arange(len(X_var))
    np.random.shuffle(indices)
    X_var = [X_var[i] for i in indices]
    y_var = [y_var[i] for i in indices]
    return X_var, y_var

def main():
    print("=" * 60)
    print("Training DFA Date String Classifier (variable-length, multi-format)")
    print("=" * 60)
    # Generate or load data
    data_path = os.path.join(MODEL_DIR, "date_train_test_split.pkl")
    
    # Remove old cache to regenerate with new formats
    if os.path.exists(data_path):
        print(f"Removing old cache: {data_path}")
        os.remove(data_path)
    
    print("Generating new date data with multiple formats (-, /)...")
    pos_data = generate_real_dates(n=15000)
    neg_data = generate_date_negatives(pos_data, n=15000)
    X = pos_data + neg_data
    y = [1]*len(pos_data) + [0]*len(neg_data)
    
    print(f"\nPositive samples (valid dates):")
    print(f"  Sample formats: {pos_data[:5]}")
    print(f"\nNegative samples (invalid dates with # and mutations):")
    print(f"  Sample formats: {neg_data[:5]}")
    
    X_seq = [list(s) for s in X]
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
    X_train = np.array(X_train, dtype=object)
    X_test = np.array(X_test, dtype=object)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    with open(data_path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }, f)
    print(f"\nTrain/test split saved to: {data_path}")
    print(f"Original training set size: {len(X_train)}")
    print(f"Original test set size: {len(X_test)}")
    # Generate variable-length training data
    print("\nGenerating variable-length training samples...")
    X_train_var, y_train_var = generate_variable_length_samples(X_train, y_train, random_state=42)
    print(f"Augmented training set size: {len(X_train_var)}")
    lengths = [len(seq) for seq in X_train_var]
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}")
    # Train classifier
    print("\nTraining classifier...")
    clf = SequenceClassifier(
        model_type='rnn',
        max_len=15,
        embedding_dim=16,
        rnn_units=64,
        num_layers=1,
        dropout=0.1,
        device='cuda'
    )
    clf.fit(X_train_var, y_train_var, epochs=30, batch_size=64)
    # Evaluate
    print("\nEvaluating on test set...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    # Save model
    model_save_path = os.path.join(MODEL_DIR, "date_classifier_trained.pth")
    clf.save(model_save_path)
    print("\n" + "=" * 60)
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)
    return clf, model_save_path

if __name__ == "__main__":
    main()
