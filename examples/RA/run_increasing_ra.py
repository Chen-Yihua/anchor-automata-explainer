"""
測試RA學習遞增序列語言
語言定義: 序列中每個元素 >= 前一個元素 (非嚴格遞增)
例如: [1.0, 2.0, 3.5, 3.5, 4.0] ✓
      [1.0, 3.0, 2.0] ✗ (3.0 -> 2.0 遞減)
"""
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
from sklearn.model_selection import train_test_split
from external_modules.interpretera.src.local_serach_synthesis.utils import generate_increasing_sequence_samples

np.random.seed(42)

# 生成遞增序列數據
num_samples = 200
max_length = 5
value_range = (0.0, 10.0)
step_range = (0.1, 1.0)

pos_samples, neg_samples = generate_increasing_sequence_samples(
    num_samples, max_length, value_range, step_range
)

X = pos_samples + neg_samples
y = [1] * len(pos_samples) + [0] * len(neg_samples)
X = np.array(X, dtype=object)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 設定
feature_names = [f'seq_{i}' for i in range(max(len(s) for s in X_train))]
categorical_names = {}
positive_indices = [i for i, label in enumerate(y_train) if label == 1]
test_instance = X_train[positive_indices[0]]

# 從訓練數據提取alphabet (常數)
all_values = set()
for seq in X_train:
    all_values.update(seq)
alphabet = sorted(list(all_values))[:10]  # 限制常數數量

print(f"Training data alphabet (constants): {alphabet[:5]}... (total: {len(alphabet)})")
print(f"Test instance: {test_instance}")

automaton_type = 'RA'

# 直接用規則函數判斷遞增序列
def predict_fn(X):
    """判斷序列是否為遞增序列"""
    preds = []
    for seq in X:
        if len(seq) <= 1:
            preds.append(1)  # 空或單元素序列視為遞增
        else:
            is_increasing = all(seq[i] <= seq[i+1] for i in range(len(seq)-1))
            preds.append(1 if is_increasing else 0)
    return np.array(preds)

# 驗證預測函數
train_acc = np.mean(predict_fn(X_train) == y_train)
test_acc = np.mean(predict_fn(X_test) == y_test)
print(f"Predictor Train acc: {train_acc:.3f}")
print(f"Predictor Test acc: {test_acc:.3f}")

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

# 執行beam search
accuracy_threshold = 0.95 
state_threshold = 3        
delta = 0.01
tau = 0.01
batch_size = 300
# coverage_samples = 1000
beam_size = 3    
max_anchor_size = None
min_samples_start = 100
n_covered_ex = 20
edit_distance = 3

with open("TestRA_increasing.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) 
    print("\n============ Training RA Explanation: Increasing Sequences ============")
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
        # coverage_samples=coverage_samples,
        batch_size=batch_size,
        min_samples_start=min_samples_start,
        n_covered_ex=n_covered_ex,
        verbose=True,
        constants=[],
        theory='Double',
        num_states=8,
        num_registers=1,
    )

    print('\n============== Initial Settings ==============')
    print(f"Explaining instance: {test_instance}")
    print(f"Prediction: {predict_fn([test_instance])[0]}")
    print('\n============== Final Result ==============')
    print('RA:', explanation.data['automata'])
    print('Training Accuracy:', explanation.data['training_accuracy'])
    print('Testing Accuracy:', explanation.data['testing_accuracy'])
    print('Number of States:', explanation.data['state'])
    
    # 測試學到的RA
    print("\n\n============== Testing Learned RA ==============")
    ra = explanation.data['automata']
    
    print("\n--- RA Structure Analysis ---")
    print(f"States: {ra.states}")
    print(f"Initial: {ra.initial_state}, Final: {ra.final_states}")
    print(f"Number of Registers: {ra.num_registers}")
    print(f"\nCondition Map (guards) - sample:")
    for cid, cond in list(ra.condition_map.items())[:10]: 
        print(f"  {cid}: {cond}")
    print(f"\nAssignment Map - sample:")
    for aid, assign in list(ra.assignment_map.items())[:10]:
        print(f"  {aid}: {assign}")
    print("=" * 60)
    
    test_cases = [
        # Positive cases (should accept) - 遞增序列
        ([1.0, 2.0, 3.0], True, "strict increasing: 1,2,3"),
        ([1.0, 1.0, 1.0], True, "all equal: 1,1,1"),
        ([1.0, 2.0, 2.0, 3.0], True, "non-strict: 1,2,2,3"),
        ([5.0], True, "single element"),
        ([0.5, 1.5, 2.5, 3.5, 4.5], True, "long increasing"),
        ([2.0, 2.0], True, "equal pair"),
        
        # Negative cases (should reject) - 非遞增
        ([3.0, 2.0, 1.0], False, "decreasing: 3,2,1"),
        ([1.0, 3.0, 2.0], False, "peak: 1,3,2"),
        ([2.0, 1.0], False, "simple decrease: 2,1"),
        ([1.0, 2.0, 3.0, 2.5], False, "drop at end: 1,2,3,2.5"),
        ([5.0, 4.0, 4.0], False, "decrease then flat: 5,4,4"),
    ]

    print(f"\n{'Test Case':<50} {'Expected':<10} {'RA Result':<12} {'Correct':<10}")
    print("=" * 90)

    correct = 0
    total = len(test_cases)

    for seq, expected, description in test_cases:
        ra_result = ra.accepts_input(seq, debug=False)
        match = "✓" if ra_result == expected else "✗ FAIL"
        
        print(f"{description:<50} {str(expected):<10} {str(ra_result):<12} {match:<10}")
        
        if ra_result == expected:
            correct += 1

    print("=" * 90)
    print(f"\nRA Test Accuracy: {correct}/{total} = {correct/total*100:.1f}%")

    if correct == total:
        print("✓ RA correctly learned increasing sequence pattern!")
        print("\n理想的RA應該:")
        print("  - 使用寄存器存儲前一個值 (r0 := curr)")
        print("  - 使用guard檢查 curr >= r0")
        print("  - 如果違反遞增條件則拒絕")
    else:
        print("✗ RA did NOT correctly learn increasing pattern")
        print(f"  Failed {total-correct}/{total} test cases")
    
    sys.stdout = sys.__stdout__

print(f"\n{'='*60}")
print(f"Results saved to: TestRA_increasing.txt")
print(f"{'='*60}")
