
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
from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from models.sequence_classifier import SequenceClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

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
    beam_size = 2
    max_anchor_size = None
    init_num_samples = 2000
    n_covered_ex = 20
    edit_distance = 10

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
        init_num_samples=init_num_samples,
        n_covered_ex=n_covered_ex,
        output_dir=output_dir,
        verbose=True,
    )
    print('\n============== Result ==============')
    print('DFA:', explanation.data['automata'])
    print('Training Accuracy:', explanation.data['training_accuracy'])
    print('Testing Accuracy:', explanation.data['testing_accuracy'])
    print('Number of States:', explanation.data['state'])

    print("\n" + "=" * 60)
    print("Training data and label:")
    for i, seq in enumerate(X_train[:10]):
        print(f"Train {i}: {seq} (Label: {y_train[i]})")
    sys.stdout = sys.__stdout__
