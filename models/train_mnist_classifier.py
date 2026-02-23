"""
Train MNIST stroke sequence classifier with variable-length sequences.
Saves the trained model for later use in run_mnist.py.
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
from sklearn.model_selection import train_test_split
from datasets.mnist_stroke_loader import load_mnist_stroke_sequences
from models.sequence_classifier import SimpleSequenceClassifier
from sklearn.metrics import accuracy_score


def generate_variable_length_samples(X, y, random_state=42):
    """
    Generate samples with variable lengths by randomly truncating sequences.
    This helps the model learn to handle different sequence lengths.
    
    For each original sequence, we generate multiple versions with different lengths.
    """
    np.random.seed(random_state)
    X_var = []
    y_var = []
    
    for seq, label in zip(X, y):
        seq_len = len(seq)
        # Generate 3 versions: original, 70% length, 50% length
        for ratio in [1.0, 0.7, 0.5]:
            new_len = max(1, int(seq_len * ratio))
            truncated_seq = seq[:new_len]
            X_var.append(truncated_seq)
            y_var.append(label)
    
    # Shuffle to mix different lengths
    indices = np.arange(len(X_var))
    np.random.shuffle(indices)
    X_var = [X_var[i] for i in indices]
    y_var = [y_var[i] for i in indices]
    
    return X_var, y_var


def main():
    print("=" * 60)
    print("Training MNIST Stroke Sequence Classifier")
    print("=" * 60)
    
    # Load dataset
    min_segments=10    
    max_segments = 15
    print(f"\nLoading MNIST stroke sequences (min_segments={min_segments}, max_segments={max_segments})...")
    X_train, X_test, y_train, y_test = load_mnist_stroke_sequences(
        discretize_mode="raw_xy",
        allow_segment=True,
        min_segments=min_segments,
        max_segments=max_segments
    )
    
    # Convert to numpy arrays
    X_train_arr = np.empty(len(X_train), dtype=object)
    X_test_arr = np.empty(len(X_test), dtype=object)
    for i, seq in enumerate(X_train):
        X_train_arr[i] = seq
    for i, seq in enumerate(X_test):
        X_test_arr[i] = seq
    X_train = X_train_arr
    X_test = X_test_arr
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"Original training set size: {len(X_train)}")
    print(f"Original test set size: {len(X_test)}")
    
    # Generate variable-length training data
    print("\nGenerating variable-length training samples...")
    X_train_var, y_train_var = generate_variable_length_samples(X_train, y_train, random_state=42)
    print(f"Augmented training set size: {len(X_train_var)}")
    print(f"Sample length distribution in augmented set:")
    lengths = [len(seq) for seq in X_train_var]
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}, Mean: {np.mean(lengths):.1f}")
    
    # Create and train classifier
    print("\nTraining classifier...")
    clf = SimpleSequenceClassifier(
        max_len=max_segments,
        embedding_dim=64,
        rnn_units=256,
        num_layers=2,
        dropout=0.3,
        device='cuda'
    )
    
    clf.fit(X_train_var, y_train_var, epochs=30, batch_size=128)
    
    # Evaluate on original test set
    print("\nEvaluating on test set...")
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model
    model_save_path = os.path.join(PROJECT_ROOT, "models", "mnist_classifier_trained.pth")
    clf.save(model_save_path)

    # Save train/test split for reproducibility
    import pickle
    split_save_path = os.path.join(PROJECT_ROOT, "models", "mnist_train_test_split.pkl")
    with open(split_save_path, "wb") as f:
        pickle.dump({
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test
        }, f)
    print(f"Train/test split saved to: {split_save_path}")

    print("\n" + "=" * 60)
    print(f"Model saved to: {model_save_path}")
    print("=" * 60)
    
    return clf, model_save_path


if __name__ == "__main__":
    main()
