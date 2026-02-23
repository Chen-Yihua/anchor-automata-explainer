
import sys, os
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

from tee import Tee
import numpy as np
from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from datasets.mnist_stroke_loader import load_mnist_stroke_sequences
from models.sequence_classifier import SimpleSequenceClassifier
from sklearn.metrics import accuracy_score

# Load train/test split from file for reproducibility
import pickle
split_save_path = os.path.join(PROJECT_ROOT, "models", "mnist_train_test_split.pkl")
if not os.path.exists(split_save_path):
    print(f"Train/test split file not found at {split_save_path}\nPlease run: python models/train_mnist_classifier.py")
    sys.exit(1)
with open(split_save_path, "rb") as f:
    split = pickle.load(f)
X_train = split["X_train"]
y_train = split["y_train"]
X_test = split["X_test"]
y_test = split["y_test"]

with open("test_result/TestMnistRPNI1.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file)

    # Load pre-trained classifier
    model_path = os.path.join(PROJECT_ROOT, "models", "mnist_classifier_trained.pth")
    
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from: {model_path}")
        clf = SimpleSequenceClassifier(device='cuda')
        clf.load(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}")
        print("Please run: python models/train_mnist_classifier.py")
        print("to train the model first.")
        sys.exit(1)

    predict_fn = lambda X: clf.predict(X)

    # get accuracy
    y_pred_train = predict_fn(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    y_pred_test = predict_fn(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    #  AnchorTabular explainer settings
    feature_names = [f'seq_{i}' for i in range(max(len(s) for s in X_train))]
    categorical_names = {}
    test_instance = X_train[18] # 18
    automaton_type = 'DFA'  # 'DFA' or 'RA'

    # all_symbols = set()
    # for seq in X_train:
    #     all_symbols.update(tuple(x) if isinstance(x, np.ndarray) else x for x in seq)
    alphabet = [(0, 1), (-1, 1), (1, 1), (1, 0), (1, -1), (-1, -1), (-1, 0), (0, -1), (0, 0)]
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
    accuracy_threshold = 0.98
    state_threshold = 5
    delta = 0.01
    tau = 0.01
    batch_size = 500
    coverage_samples = 1000
    beam_size = 2
    max_anchor_size = None
    min_samples_start = 1000
    # n_covered_ex = 20
    edit_distance = 5

    print(f"\nExplaining instance: {test_instance}")
    print(f"Prediction: {predict_fn([test_instance])}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\n============ Training RA Explanation (mnist) ============")
    explanation = explainer.explain(
        type='Tabular',
        automaton_type=automaton_type,
        alphabet=alphabet,
        X=test_instance,
        edit_distance=edit_distance,
        accuracy_threshold=accuracy_threshold,
        state_threshold=state_threshold,
        delta=delta,
        tau=tau,
        beam_size=beam_size,
        max_anchor_size=max_anchor_size,
        coverage_samples=coverage_samples,
        batch_size=batch_size,
        min_samples_start=min_samples_start,
        # n_covered_ex=n_covered_ex,
        verbose=True,
    )
    print('\n============== Result ==============')
    print('RA:', explanation.data['automata'])
    print('Training Accuracy:', explanation.data['training_accuracy'])
    print('Testing Accuracy:', explanation.data['testing_accuracy'])
    print('Number of States:', explanation.data['state'])
    sys.stdout = sys.__stdout__