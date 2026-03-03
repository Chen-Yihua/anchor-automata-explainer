"""
Train Tomita language sequence classifier for DFA experiments.
Saves the trained model and train/test split for later use in run_tomita.py.
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

import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.sequence_classifier import SequenceClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'datasets')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

# Import Tomita DFA language generators
from datasets.tomita_loader_dfa import (
    L3AB, L4, L5, L6, L7
)

LANGUAGES = {
    # "L3AB": L3AB,
    # "L4": L4,
    # "L5": L5,
    "L6": L6,
    # "L7": L7,
}


def generate_and_train(lang_code, lang_class):
    print(f"\n{'='*60}")
    print(f"Training Tomita language: {lang_code}")
    print(f"{'='*60}")

    # Set hyperparameters for each language
    if lang_code == 'L3AB':
        num_pos=5000
        num_neg=5000
        max_length=10
        embedding_dim = 16
        hidden_dim = 32
        num_layers=1
        dropout = 0
        weight_decay = 1e-4
        epochs = 10
        batch_size = 32
        use_attention = False
    elif lang_code == 'L4':
        num_pos=5000
        num_neg=5000
        max_length=15
        embedding_dim = 16
        hidden_dim = 32
        num_layers=1
        dropout = 0
        weight_decay = 1e-4
        epochs = 10
        batch_size = 32
        use_attention = False
    elif lang_code == 'L5':
        num_pos=5000
        num_neg=5000
        max_length=10
        embedding_dim = 16
        hidden_dim = 32
        num_layers=1
        dropout = 0
        weight_decay = 1e-4
        epochs = 10
        batch_size = 32
        use_attention = False
    elif lang_code == 'L6':
        num_pos=5000
        num_neg=5000
        max_length=10
        embedding_dim = 16
        hidden_dim = 32
        num_layers=1
        dropout = 0.5
        weight_decay = 5e-4
        epochs = 20
        batch_size = 32
        use_attention = False
    else:
        num_pos=5000
        num_neg=5000
        max_length=10
        embedding_dim = 16
        hidden_dim = 32
        num_layers=1
        dropout = 0
        weight_decay = 1e-4
        epochs = 20
        batch_size = 32
        use_attention = False

    # Generate Data
    data_gen_start = time.time()
    model = lang_class()
    pos_samples, neg_samples = model.generate_samples(num_pos=num_pos, num_neg=num_neg, max_length=max_length)
    data_gen_time = time.time() - data_gen_start
    print(f"Data generation time: {data_gen_time:.2f}s")

    X = pos_samples + neg_samples
    y = [1] * len(pos_samples) + [0] * len(neg_samples)
    X = np.array(X, dtype=object)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train classifier
    print("\nTraining classifier...")
    
    clf = SequenceClassifier(
        model_type='rnn',
        max_len=max_length,
        embedding_dim=embedding_dim,
        rnn_units=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        weight_decay=weight_decay,
        use_attention=use_attention,
        device='cuda'
    )
    clf.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model
    model_save_path = os.path.join(MODEL_DIR, f"{lang_code}_classifier_trained.pth")
    clf.save(model_save_path)
    print(f"Model saved to: {model_save_path}")

    # Save train/test split for reproducibility
    import pickle
    split_save_path = os.path.join(MODEL_DIR, f"{lang_code}_train_test_split.pkl")
    with open(split_save_path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }, f)
    print(f"Train/test split saved to: {split_save_path}")
    print(f"{'='*60}")
    return clf, model_save_path

def main():
    for lang_code, lang_class in LANGUAGES.items():
        generate_and_train(lang_code, lang_class)

if __name__ == "__main__":
    main()
