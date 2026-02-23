import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

for path in [SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from tee import Tee
import numpy as np
from modified_modules.alibi.explainers import AnchorTabular
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from external_modules.interpretera.src.local_serach_synthesis.utils import generate_nominal_language_samples, generate_increasing_sequence_samples

np.random.seed(0)


num_samples = 200
max_length = 5
value_range = (0.0, 10.0)
step_range = (0.1, 1.0)
pos_samples, neg_samples = generate_nominal_language_samples(num_samples, max_length, value_range)

X = pos_samples + neg_samples
y = [1] * len(pos_samples) + [0] * len(neg_samples)
X = np.array(X, dtype=object)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  AnchorTabular explainer 設定
feature_names = [f'seq_{i}' for i in range(max(len(s) for s in X_train))]
categorical_names = {}
positive_indices = [i for i, label in enumerate(y_train) if label == 1]
test_instance = X_train[positive_indices[0]]
alphabet = sorted(set(test_instance))
alphabet.append(5.0) # 確保有多個常數

print(f"Training data alphabet (constants): {alphabet}")
automaton_type = 'RA'  # 'DFA' or 'RA'

def feature_fn(seq):
    if len(seq) == 0:
        return [0, 1]  # 空字串視為合法
    
    if len(seq) < 1:
        return [len(seq), 0]
    
    # a* language: all elements should be the same
    first_val = seq[0]
    is_a_star = all(val == first_val for val in seq)
    
    return [len(seq), int(is_a_star)]

X_train_feat = np.array([feature_fn(seq) for seq in X_train])
X_test_feat = np.array([feature_fn(seq) for seq in X_test])

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_feat, y_train)

def predict_fn(X):
    feats = np.array([feature_fn(seq) for seq in X])
    return clf.predict(feats)

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

# 跑 explainer.explain (beam search)
accuracy_threshold = 0.9
state_threshold = 3
delta = 0.1
tau = 0.15
batch_size = 100
coverage_samples = 1000
beam_size = 3
max_anchor_size = None
min_samples_start = 100
n_covered_ex = 20
edit_distance = 2

with open("TestRA_ab.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) 
    print(f"\nExplaining instance: {test_instance}")
    print(f"Prediction: {predict_fn([test_instance])[0]}")
    print("Model Train acc:", clf.score(X_train_feat, y_train))
    print("Model Test acc :", clf.score(X_test_feat, y_test))
    
    print("\n============ Training RA Explanation a* ============")
    explanation = explainer.explain(
        type='Tabular',
        automaton_type=automaton_type,
        alphabet=alphabet,
        X=np.array([test_instance], dtype=object),
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
        n_covered_ex=n_covered_ex,
        verbose=True,
    )
    print('\n============== Result ==============')
    print('RA:', explanation.data['automata'])
    print('Training Accuracy:', explanation.data['training_accuracy'])
    print('Testing Accuracy:', explanation.data['testing_accuracy'])
    print('Number of States:', explanation.data['state'])
    
    # Test the learned RA on specific a* test cases
    print("\n\n============== Testing Learned RA ==============")
    ra = explanation.data['automata']
    
    print("\n--- RA Structure Analysis ---")
    print(f"States: {ra.states}")
    print(f"Initial: {ra.initial_state}, Final: {ra.final_states}")
    print(f"\nCondition Map (guards):")
    for cid, cond in list(ra.condition_map.items())[:10]: 
        print(f"  {cid}: {cond}")
    print(f"\nTransitions from state 0:")
    if 0 in ra.transitions:
        for i, (cond_id, dst) in enumerate(ra.transitions[0]):
            guard = ra.condition_map.get(cond_id, "?")
            print(f"  -> state {dst}: guard={guard} (id={cond_id})")
            if 0 in ra.register_transitions and i < len(ra.register_transitions[0]):
                assign_id, _ = ra.register_transitions[0][i]
                assign = ra.assignment_map.get(assign_id, "?")
                print(f"     assignment: {assign}")
    print("=" * 60)
    
    a_val = alphabet[0] if alphabet else 5.0
    test_cases = [
        # Positive cases (should accept) - all same value
        ([a_val, a_val, a_val, a_val], True, "aaaa"),
        ([a_val], True, "a"),
        ([a_val, a_val], True, "aa"),
        ([a_val, a_val, a_val, a_val, a_val], True, "aaaaa"),
        
        # Negative cases (should reject) - mixed values
        ([a_val, a_val + 1], False, "ab - different values"),
        ([a_val, a_val, a_val + 0.5], False, "aac - last different"),
        ([a_val + 1, a_val], False, "ba - first different"),
        ([a_val, a_val + 1, a_val], False, "aba - middle different"),
        ([1.0, 2.0, 3.0], False, "abc - all different"),
    ]

    print(f"\n{'Test Case':<40} {'Expected':<10} {'RA Result':<12} {'Correct':<10}")
    print("=" * 80)

    correct = 0
    total = len(test_cases)

    for seq, expected, description in test_cases:
        ra_result = ra.accepts_input(seq, debug=False)
        match = "✓" if ra_result == expected else "✗ FAIL"
        
        print(f"{description:<40} {str(expected):<10} {str(ra_result):<12} {match:<10}")
        
        if ra_result == expected:
            correct += 1

    print("=" * 80)
    print(f"\nRA Test Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    if correct == total:
        print("✓ RA correctly learned a* pattern!")
    else:
        print("✗ RA did NOT correctly learn a* pattern - may be overfitting training data")
    
    sys.stdout = sys.__stdout__