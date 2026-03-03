
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

# from models.simple_sequence_classifier import SimpleSequenceClassifier
from models.sequence_classifier import SequenceClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- Language Models to Benchmark ---
LANGUAGES = {
    # "L1 (Constant Sequence)": L1,
    # "L2 (Pattern (abcd)*)": L2,
    # "L2A (Pattern (aa)*)": L2A,
    # "L2AB (Pattern (abcdabcd)*)": L2AB,
    # "L3 (Parity of counts)": L3,
    # "L3A (Pattern (aaa)*)": L3A,
    # "L3AB (Pattern (abcdabcdabcd)*)": L3AB, # 30 states: ['a', 'b', 'c', 'd'] * 6 -
    # "L4 (No three consecutive same)": L4,
    # "L4A (Pattern (aaaa)*)": L4A,
    # "L4AB (Pattern (abcdabcdabcdabcd)*)": L4AB,
    # "L5 (Even count of A and B)": L5,
    # "L5A (Pattern (aaaaa)*)": L5A,
    # "L5AB (Pattern (abcdabcdabcdabcdabcd)*)": L5AB,
    "L6 (|#a - #b| mod 3 == 0)": L6,
    # "L6A (Pattern (aaaaaa)*)": L6A,
    # "L7 (Pattern a*b*a*b*)": L7,
    # "L7A (Pattern (aaaaaaa)*)": L7A,
    # "L8A (Pattern (aaaaaaaa)*)": L8A,
    # "L9A (Pattern (aaaaaaaaa)*)": L9A,
    # "L10A (Pattern (aaaaaaaaaa)*)": L10A,
}

for lang_name, lang_class in LANGUAGES.items():
    lang_code = lang_name.split(" ")[0]
    
    # Create output directory for this language
    output_dir = f"test_result/tomita_{lang_code}"
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"tomita_{lang_code}.txt")
    
    with open(txt_path, "w", encoding="utf-8") as log_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)

        print(f"\n{'='*80}")
        print(f"Running tomita for: {lang_name}")
        print(f"{'='*80}\n")
        
        # Start timing
        total_start_time = time.time()

        # set hyperparameters based on language
        if lang_code == 'L3AB':
            num_pos=5000
            num_neg=5000
            max_length=10
            embedding_dim = 16
            hidden_dim = 32
            num_layers=1
            dropout = 0
            epochs = 20
            batch_size = 32
            test_instance = ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a', 'b', 'c', 'd']
            alphabet = ['a', 'b', 'c', 'd']
            accuracy_threshold = 0.9
            state_threshold = 5
            delta = 0.01
            tau = 0.01
            batch_size = 2000
            coverage_samples = 1000
            beam_size = 1
            init_num_samples = 2000
            edit_distance = 4
        elif lang_code == 'L4':
            num_pos=10000
            num_neg=10000
            max_length=10
            embedding_dim = 16
            hidden_dim = 64
            num_layers=2
            dropout = 0.5
            epochs = 30
            batch_size = 128
            test_instance = ['a', 'a', 'b', 'a', 'c', 'c', 'd', 'b', 'b'] 
            alphabet = ['a', 'b', 'c', 'd']
            accuracy_threshold = 0.9
            state_threshold = 5
            delta = 0.01
            tau = 0.01
            batch_size = 500
            coverage_samples = 1000
            beam_size = 1
            init_num_samples = 1000
            edit_distance = 4
        # elif lang_code == 'L5':
        #     num_pos=5000
        #     num_neg=5000
        #     max_length=10
        #     embedding_dim = 16
        #     hidden_dim = 32
        #     num_layers=1
        #     dropout = 0
        #     epochs = 10
        #     batch_size = 32
        #     use_attention = False
        #     test_instance = ['b', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'b']
        #     alphabet = ['a', 'b']
        #     accuracy_threshold = 0.95
        #     state_threshold = 5
        #     delta = 0.01
        #     tau = 0.01
        #     batch_size = 500
        #     coverage_samples = 1000
        #     beam_size = 1
        #     init_num_samples = 2000
        #     edit_distance = 8
        elif lang_code == 'L6':
            num_pos=5000
            num_neg=5000
            max_length=10
            embedding_dim = 16
            hidden_dim = 32
            num_layers=1
            dropout = 0.5
            epochs = 20
            batch_size = 32
            use_attention = False
            test_instance = ['a', 'a', 'a', 'c', 'a', 'a', 'b', 'b']
            alphabet = ['a', 'b', 'c', 'd']
            accuracy_threshold = 0.9
            state_threshold = 5
            delta = 0.01
            tau = 0.01
            batch_size = 500
            coverage_samples = 1000
            beam_size = 1
            init_num_samples = 2000
            edit_distance = 3
        else:  # L7
            num_pos=5000
            num_neg=5000
            max_length=10
            embedding_dim = 16
            hidden_dim = 32
            num_layers=1
            dropout = 0
            epochs = 20
            batch_size = 32
            use_attention = False
            test_instance = ['a', 'a', 'b', 'b', 'b', 'a', 'a', 'b', 'b', 'b']
            alphabet = ['a', 'b', 'c', 'd']
            accuracy_threshold = 0.9
            state_threshold = 5
            delta = 0.01
            tau = 0.01
            batch_size = 500
            coverage_samples = 1000
            beam_size = 1
            init_num_samples = 2000
            edit_distance = 10

        # Load train/test split
        import pickle
        split_save_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_train_test_split.pkl")
        if not os.path.exists(split_save_path):
            print(f"Train/test split file not found at {split_save_path}\nPlease run: python models/train_tomita_classifier.py")
            sys.exit(1)
        with open(split_save_path, "rb") as f:
            split = pickle.load(f)
        X_train = split["X_train"]
        y_train = split["y_train"]
        X_test = split["X_test"]
        y_test = split["y_test"]

        # Load pre-trained classifier
        model_path = os.path.join(PROJECT_ROOT, "models", f"{lang_code}_classifier_trained.pth")
        if os.path.exists(model_path):
            print(f"Loading pre-trained model from: {model_path}")
            clf = SequenceClassifier(max_len=max_length, embedding_dim=embedding_dim, device='cuda')
            clf.load(model_path)
            print("Model loaded successfully!")
        else:
            print(f"Model not found at {model_path}")
            print(f"Please run: python models/train_tomita_classifier.py")
            sys.exit(1)
        predict_fn = lambda X: clf.predict(X)

        y_pred_train = predict_fn(X_train)
        train_acc = accuracy_score(y_train, y_pred_train)
        y_pred_test = predict_fn(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        explainer = AnchorTabular(
            predictor=predict_fn,
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
        # accuracy_threshold = 0.9
        # state_threshold = 5
        # delta = 0.01
        # tau = 0.01
        # batch_size = 500
        # coverage_samples = 1000
        # beam_size = 2
        # init_num_samples = 2000
        # edit_distance = 5
        
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
            init_num_samples=init_num_samples,
            # n_covered_ex=n_covered_ex,
            verbose=True,
            constants=[],
            output_dir=output_dir,
        )
        dfa_learning_time = time.time() - dfa_start_time
        total_time = time.time() - total_start_time

        # 5. Print Results
        print('\n============== Final Result ==============')
        print(f"Explaining instance: {test_instance}")
        print(f"Prediction: {predict_fn([test_instance])}")
        
        if explanation and explanation.data:
            dfa = explanation.data.get('automata', None)
            print('DFA:', dfa)
            print('Training Accuracy (擾動樣本):', explanation.data.get('training_accuracy', 'N/A'))
            print('Testing Accuracy (擾動樣本):', explanation.data.get('testing_accuracy', 'N/A'))
            print('Number of States:', explanation.data.get('state', 'N/A'))
        else:
            print("Explanation failed or did not produce a valid automaton.")

        print("\n" + "=" * 60)
        print("Training data and label:")
        for i, seq in enumerate(X_train[:10]):
            print(f"Train {i}: {seq} (Label: {y_train[i]})")
        
        # Print timing information
        print(f"\n============== Timing ==============")
        # print(f"Data generation time: {data_gen_time:.2f}s")
        print(f"DFA learning time: {dfa_learning_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

    sys.stdout = original_stdout
