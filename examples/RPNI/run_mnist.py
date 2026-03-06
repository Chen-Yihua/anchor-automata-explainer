
import sys, os
import time
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
from models.sequence_classifier import SequenceClassifier
from sklearn.metrics import accuracy_score
import pickle
from learner import AUTO_INSTANCE

output_dir = "test_result/TestMnistRPNI_2"
os.makedirs(output_dir, exist_ok=True)
txt_path = os.path.join(output_dir, "TestMnistRPNI_2.txt")
with open(txt_path, "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file)

    # Load train/test split
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

    # Load pre-trained classifier
    model_path = os.path.join(PROJECT_ROOT, "models", "mnist_classifier_trained.pth")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from: {model_path}")
        clf = SequenceClassifier(device='cuda')
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
    test_instance = X_train[57] # 23(7)、57(2)
    automaton_type = 'DFA'  # 'DFA' or 'RA'

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
    select_by = 'accuracy'
    accuracy_threshold = 0.8
    state_threshold = 5
    delta = 0.05
    tau = 0.05
    batch_size = 2000
    coverage_samples = 1000
    beam_size = 1
    max_anchor_size = None
    init_num_samples = 1000
    edit_distance = 10

    print("\n============ Training DFA Explanation (mnist) ============")
    start_time = time.time()
    explanation = explainer.explain(
        type='Tabular',
        automaton_type=automaton_type,
        alphabet=alphabet,
        X=test_instance,
        edit_distance=edit_distance,
        accuracy_threshold=accuracy_threshold,
        select_by=select_by,
        state_threshold=state_threshold,
        delta=delta,
        tau=tau,
        beam_size=beam_size,
        # max_anchor_size=max_anchor_size,
        # coverage_samples=coverage_samples,
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
    print("Testing Final Automaton on Training Data")
    print("=" * 60)
    
    target_label = predict_fn([test_instance])[0]
    
    print(f"\nTarget digit: {target_label}")
    print(f"Test instance: {test_instance}\n")
    
    np.random.seed(42)
    target_seqs = [seq for seq, label in zip(X_test, y_test) if label == target_label]
    other_seqs = [seq for seq, label in zip(X_test, y_test) if label != target_label]
    
    if len(target_seqs) > 0:
        sample_target_idx = np.random.choice(len(target_seqs), min(5, len(target_seqs)), replace=False)
        target_accept = sum(1 for i in sample_target_idx if AUTO_INSTANCE.check_path_accepted(automaton, target_seqs[i]))
        print(f"Digit {target_label} - Acceptance: {target_accept}/{len(sample_target_idx)}")
        for i in sample_target_idx[:10]:
            accepted = AUTO_INSTANCE.check_path_accepted(automaton, target_seqs[i])
            print(f"  {'Accept' if accepted else 'Reject'}: {target_seqs[i]}")
    
    if len(other_seqs) > 0:
        sample_other_idx = np.random.choice(len(other_seqs), min(5, len(other_seqs)), replace=False)
        other_reject = sum(1 for i in sample_other_idx if not AUTO_INSTANCE.check_path_accepted(automaton, other_seqs[i]))
        print(f"\nOther digits - Rejection: {other_reject}/{len(sample_other_idx)}")
        for i in sample_other_idx[:10]:
            seq = other_seqs[i]
            label = next(y for s, y in zip(X_test, y_test) if s == seq)
            accepted = AUTO_INSTANCE.check_path_accepted(automaton, seq)
            print(f"  {'Reject' if not accepted else 'Accept'} (Digit {label}): {seq}")
    
    # Print timing information
    print(f"\n============== Timing ==============")
    print(f"DFA learning time: {learning_time:.2f}s")
    sys.stdout = sys.__stdout__