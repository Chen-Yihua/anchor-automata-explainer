
import sys, os
import time
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
from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from models.sequence_classifier import SequenceClassifier
from sklearn.metrics import accuracy_score
import pickle
from learner import AUTO_INSTANCE
import numpy as np

output_dir = "test_result/TestDFA_date_fixed_length"
os.makedirs(output_dir, exist_ok=True)
txt_path = os.path.join(output_dir, "TestDFA_date_fixed_length.txt")
with open(txt_path, "w", encoding="utf-8") as log_file:
    tee = Tee(sys.__stdout__, log_file)
    sys.stdout = tee

    # Load train/test split
    split_save_path = os.path.join(PROJECT_ROOT, "models", "date_train_test_split.pkl")
    if not os.path.exists(split_save_path):
        print(f"Train/test split file not found at {split_save_path}\nPlease run: python models/train_dfa_date_classifier.py")
        sys.exit(1)
    with open(split_save_path, "rb") as f:
        split = pickle.load(f)
    X_train = split["X_train"]
    y_train = split["y_train"]
    X_test = split["X_test"]
    y_test = split["y_test"]

    # Load pre-trained classifier
    model_path = os.path.join(PROJECT_ROOT, "models", "date_classifier_trained.pth")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from: {model_path}")
        clf = SequenceClassifier(max_len=15, embedding_dim=16, device='cuda')
        clf.load(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}")
        print("Please run: python models/train_date_classifier.py")
        sys.exit(1)
    predict_fn = lambda X: clf.predict(X)

    y_pred_train = predict_fn(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    y_pred_test = predict_fn(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)

    # AnchorTabular explainer setup
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
    accuracy_threshold = 0.9
    state_threshold = 8
    select_by = 'accuracy'  # 'accuracy': 達標中選最小狀態; 'state': 狀態數限制內選最高精確度
    delta = 0.01
    tau = 0.01
    batch_size = 2000
    coverage_samples = 1000
    beam_size = 1
    max_anchor_size = None
    init_num_samples = 2000
    edit_distance = 10

    print("\n============ Training DFA Explanation (MM-DD-YYYY) ============")
    start_time = time.time()
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
        init_num_samples=init_num_samples,
        output_dir=output_dir,
        verbose=True,
    )
    learning_time = time.time() - start_time
    automaton = explanation.data['automata']

    print('\n============== Result ==============')
    print(f"Training Accuracy of model: {train_acc:.4f}")
    print(f"Testing Accuracy of model: {test_acc:.4f}")
    print(f"Explaining Instance: {test_instance}")
    print(f"Prediction of Instance: {predict_fn([test_instance])}")

    print('\nFinal DFA:', automaton)
    print('Number of States:', explanation.data['state'])
    print('Training Accuracy of DFA:', explanation.data['training_accuracy'])
    print('Testing Accuracy of DFA:', explanation.data['testing_accuracy'])

    print("\n" + "=" * 60)
    print("Quick Validation - Sample Test")
    print("=" * 60)
    
    automaton = explanation.data['automata']
    target_label = predict_fn([test_instance])[0]
    
    print(f"\nTarget label: {target_label}")
    print(f"Test instance: {test_instance}\n")
    
    np.random.seed(42)
    target_seqs = [seq for seq, label in zip(X_test, y_test) if label == target_label]
    other_seqs = [seq for seq, label in zip(X_test, y_test) if label != target_label]
    
    if len(target_seqs) > 0:
        sample_target_idx = np.random.choice(len(target_seqs), min(5, len(target_seqs)), replace=False)
        target_accept = sum(1 for i in sample_target_idx if AUTO_INSTANCE.check_path_accepted(automaton, target_seqs[i]))
        print(f"Target label {target_label} - Acceptance: {target_accept}/{len(sample_target_idx)}")
        for i in sample_target_idx[:10]:
            accepted = AUTO_INSTANCE.check_path_accepted(automaton, target_seqs[i])
            print(f"  {'Accept' if accepted else 'Reject'}: {''.join(target_seqs[i])}")
    
    if len(other_seqs) > 0:
        sample_other_idx = np.random.choice(len(other_seqs), min(5, len(other_seqs)), replace=False)
        other_reject = sum(1 for i in sample_other_idx if not AUTO_INSTANCE.check_path_accepted(automaton, other_seqs[i]))
        print(f"\nOther labels - Rejection: {other_reject}/{len(sample_other_idx)}")
        for i in sample_other_idx[:10]:
            seq = other_seqs[i]
            label = next(y for s, y in zip(X_test, y_test) if s == seq)
            accepted = AUTO_INSTANCE.check_path_accepted(automaton, seq)
            print(f"  {'Reject' if not accepted else 'Accept'} (Label {label}): {''.join(seq)}")

    print("\n" + "=" * 60)
    print("Training data and label:")
    for i, seq in enumerate(X_train[:10]):
        print(f"Train {i}: {seq} (Label: {y_train[i]})")

    # Print timing information
    print(f"\n============== Timing ==============")
    print(f"DFA learning time: {learning_time:.2f}s")
    sys.stdout = sys.__stdout__
