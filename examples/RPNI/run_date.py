
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

for path in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from src.tee import Tee
import numpy as np
from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
# from models.binary_sequence_classifier import SimpleBinarySequenceClassifier
from models.sequence_classifier import SimpleSequenceClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import datetime
import random as _random

# def generate_real_dates(n=3000):
#     """用 datetime 產生 n 個合法日期字串 MM-DD-YYYY"""
#     dates = set()
#     start = datetime.date(1900, 1, 1)
#     end = datetime.date(2026, 12, 31)
#     delta_days = (end - start).days
#     while len(dates) < n:
#         rand_day = start + datetime.timedelta(days=_random.randint(0, delta_days))
#         dates.add(rand_day.strftime("%m-%d-%Y"))  # MM-DD-YYYY
#     return list(dates)

# def generate_date_negatives(positive_samples, n=3000):
#     """對合法日期做擾動，產生負例（只用 0-9 和 - 的字元）"""
#     alphabet = sorted(set(c for s in positive_samples for c in s))  # ['0'-'9', '-']
#     negatives = set()
#     fmt_pattern = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    
#     def is_valid_date(s):
#         if not fmt_pattern.fullmatch(s):
#             return False
#         try:
#             datetime.datetime.strptime(s, "%m-%d-%Y")
#             return True
#         except ValueError:
#             return False
    
#     attempts = 0
#     while len(negatives) < n and attempts < n * 20:
#         base = _random.choice(positive_samples)
#         char_list = list(base)
#         mutation = _random.choice(['delete', 'insert', 'replace', 'swap'])
#         idx = _random.randint(0, len(char_list) - 1)
        
#         if mutation == 'delete' and len(char_list) > 1:
#             del char_list[idx]
#         elif mutation == 'insert':
#             char_list.insert(idx, _random.choice(alphabet))
#         elif mutation == 'replace':
#             original = char_list[idx]
#             candidates = [c for c in alphabet if c != original]
#             if candidates:
#                 char_list[idx] = _random.choice(candidates)
#         elif mutation == 'swap' and len(char_list) > 1:
#             j = _random.randint(0, len(char_list) - 1)
#             char_list[idx], char_list[j] = char_list[j], char_list[idx]
        
#         mutated = ''.join(char_list)
#         if not is_valid_date(mutated):
#             negatives.add(mutated)
#         attempts += 1
    
#     # fallback: 用 alphabet 隨機組字串
#     while len(negatives) < n:
#         length = _random.randint(8, 12)
#         rand_str = ''.join(_random.choice(alphabet) for _ in range(length))
#         if not is_valid_date(rand_str):
#             negatives.add(rand_str)
#     return list(negatives)

import re

# Load train/test split from datasets/dfa_date_train_test_split.pkl
import pickle
split_save_path = os.path.join(PROJECT_ROOT, "datasets", "dfa_date_train_test_split.pkl")
if not os.path.exists(split_save_path):
    print(f"Train/test split file not found at {split_save_path}\nPlease run: python models/train_dfa_date_classifier.py")
    sys.exit(1)
with open(split_save_path, "rb") as f:
    split = pickle.load(f)
X_train = split["X_train"]
y_train = split["y_train"]
X_test = split["X_test"]
y_test = split["y_test"]


with open("test_result/TestDFA_date_fixed_length.txt", "w", encoding="utf-8") as log_file:
    tee = Tee(sys.__stdout__, log_file)
    sys.stdout = tee

    # Load pre-trained classifier
    model_path = os.path.join(PROJECT_ROOT, "models", "dfa_date_classifier_trained.pth")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from: {model_path}")
        clf = SimpleSequenceClassifier(max_len=15, embedding_dim=16, rnn_units=64, num_layers=1, dropout=0.1, device='cuda')
        clf.load(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}")
        print("Please run: python models/train_dfa_date_classifier.py")
        sys.exit(1)
    predict_fn = lambda X: clf.predict(X)

    y_pred_train = predict_fn(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    y_pred_test = predict_fn(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    # AnchorTabular explainer 設定
    feature_names = [f'char_{i}' for i in range(max(len(s) for s in X_train))]
    categorical_names = {}
    positive_indices = [i for i, label in enumerate(y_train) if label == 1]
    test_instance = X_train[positive_indices[2]]
    alphabet = sorted(set(c for seq in X_train for c in seq))
    automaton_type = 'DFA'

    explainer = AnchorTabular(
        predictor=predict_fn,
        feature_names=feature_names,
        categorical_names=categorical_names,
        seed=1
    )

    explainer.fit(
        automaton_type=automaton_type,
        train_data=X_train,
        alphabet=alphabet,
        disc_perc=None
    )
    explainer.samplers[0].d_train_data = X_train

    # run explainer.explain (beam search)
    accuracy_threshold = 0.95
    state_threshold = 8
    select_by = 'accuracy'  # 'accuracy': 達標中選最小狀態; 'state': 狀態數限制內選最高精確度
    delta = 0.01
    tau = 0.01
    batch_size = 1000
    coverage_samples = 1000
    beam_size = 3
    max_anchor_size = None
    min_samples_start = 3000
    n_covered_ex = 20
    edit_distance = 5

    print(f"\nExplaining instance: {test_instance}")
    print(f"Prediction: {predict_fn([test_instance])[0]}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\n============ Training DFA Explanation (MM-DD-YYYY) ============")
    explanation = explainer.explain(
        type='Tabular',
        automaton_type=automaton_type,
        alphabet=alphabet,
        X=test_instance,
        edit_distance=edit_distance,
        accuracy_threshold=accuracy_threshold,
        state_threshold=state_threshold,
        select_by=select_by,
        delta=delta,
        tau=tau,
        beam_size=beam_size,
        max_anchor_size=max_anchor_size,
        coverage_samples=coverage_samples,
        batch_size=batch_size,
        min_samples_start=min_samples_start,
        n_covered_ex=n_covered_ex,
        verbose=True,
    )
    print('\n============== Result ==============')
    print('DFA:', explanation.data['automata'])
    print('Training Accuracy:', explanation.data['training_accuracy'])
    print('Testing Accuracy:', explanation.data['testing_accuracy'])
    print('Number of States:', explanation.data['state'])
    sys.stdout = sys.__stdout__
