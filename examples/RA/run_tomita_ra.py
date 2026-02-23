"""
Benchmark RA learning on different sequence languages defined in benchmark_improved.py.
Languages: S1, S2, L2, L3, L6
"""
import sys, os
import time
import numpy as np
from sklearn.model_selection import train_test_split

# --- Project Paths Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

for path in [SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)

from modified_modules.alibi.explainers.anchors.anchor_tabular import AnchorTabular
from tee import Tee
from datasets.benchmark_improved import (
    S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14,
    L1, L2, L2A, L3, L3A, L4, L4A, L5, L5A, L6, L6A, L7, L7A, L8A, L9A, L10A,
    L2AB, L3AB, L4AB, L5AB
)

np.random.seed(42)

# --- Language Models to Benchmark ---
LANGUAGES = {
    # "S1 (Strictly Increasing)": S1,
    # "S2 (Strictly Decreasing)": S2,
    # "S3 (Weakly Increasing)": S3, # 重跑
    # "S4 (Weakly Decreasing)": S4,
    # "S5 (Up then Down)": S5,
    # "S6 (Down then Up)": S6,
    # "S7 (Up-Down-Up Prefix)": S7,
    # "S8 (Up-Down-Up-Down Prefix)": S8,
    # "S9 (Higher Peaks, Higher Troughs)": S9, 
    # "S10 (Higher Peaks, Lower Troughs)": S10,
    # "S11 (Lower Peaks, Lower Troughs)": S11,
    # "S12 (5-Alt Prefix)": S12,
    # "S13 (6-Alt Prefix)": S13, # 重跑
    # "S14 (7-Alt Prefix)": S14, # 重跑
    "L1 (Constant Sequence)": L1,
    "L2 (Pattern (ab)*)": L2,
    "L2A (Pattern (aa)*)": L2A,
    "L3 (Parity of counts)": L3,
    "L3A (Pattern (aaa)*)": L3A,
    # "L4 (No three consecutive same)": L4,
    # "L4A (Pattern (aaaa)*)": L4A,
    # "L5 (Even count of A and B)": L5,
    # "L5A (Pattern (aaaaa)*)": L5A,
    # "L6 (Modulo Balance)": L6,
    # "L6A (Pattern (aaaaaa)*)": L6A,
    # "L7 (Pattern a*b*a*b*)": L7,
    # "L7A (Pattern (aaaaaaa)*)": L7A,
    # "L8A (Pattern (aaaaaaaa)*)": L8A,
    # "L9A (Pattern (aaaaaaaaa)*)": L9A,
    # "L10A (Pattern (aaaaaaaaaa)*)": L10A,
    # "L2AB (Pattern (abab)*)": L2AB,
    # "L3AB (Pattern (ababab)*)": L3AB,
    # "L4AB (Pattern (abababab)*)": L4AB,
    # "L5AB (Pattern (ababababab)*)": L5AB,
}

# --- Main Loop ---
for lang_name, lang_class in LANGUAGES.items():
    lang_code = lang_name.split(" ")[0]
    
    # Create output directory for this language
    output_dir = os.path.join(PROJECT_ROOT, "test_result", f"benchmark_{lang_code}")
    os.makedirs(output_dir, exist_ok=True)
    
    output_filename = os.path.join(output_dir, f"benchmark_{lang_code}.txt")
    with open(output_filename, "w", encoding="utf-8") as log_file:
        # Redirect stdout to both console and file
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_file)

        print(f"\n{'='*80}")
        print(f"Running Benchmark for: {lang_name}")
        print(f"{'='*80}\n")
        
        # Start timing
        total_start_time = time.time()

        # 1. Generate Data
        data_gen_start = time.time()
        model = lang_class()
        pos_samples, neg_samples = model.generate_samples(num_pos=200, num_neg=200, max_length=10)

        X = pos_samples + neg_samples
        y = [1] * len(pos_samples) + [0] * len(neg_samples)
        X = np.array(X, dtype=object)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        data_gen_time = time.time() - data_gen_start
        print(f"Data generation time: {data_gen_time:.2f}s")

        # 2. Define Predictor
        def predict_fn(data):
            return np.array([1 if model.check(seq) else 0 for seq in data])

        train_acc = np.mean(predict_fn(X_train) == y_train)
        test_acc = np.mean(predict_fn(X_test) == y_test)
        print(f"Predictor Train Accuracy: {train_acc:.3f}")
        print(f"Predictor Test Accuracy: {test_acc:.3f}")

        # 3. Setup Explainer
        positive_indices = [i for i, label in enumerate(y_train) if label == 1]
        if not positive_indices:
            print("No positive samples in training set, skipping explanation.")
            continue
        test_instance = X_train[positive_indices[0]]

        # ============================================================
        # Alphabet：實例範圍內 + 外部一點點
        # ============================================================
        instance_values = list(test_instance)
        
        # 判斷是否為數值型數據
        is_numeric = all(isinstance(v, (int, float, np.integer, np.floating)) for v in instance_values)
        
        if is_numeric:
            # 數值型數據：使用實例範圍 + 外部一點
            inst_min = int(min(instance_values))
            inst_max = int(max(instance_values))
            
            # 外部擴展：實例範圍的 20%，至少 3
            margin = max(3, (inst_max - inst_min) // 5)
            alpha_min = inst_min - margin
            alpha_max = inst_max + margin
            
            # 產生連續整數 alphabet（範圍內所有整數）
            alphabet = list(range(alpha_min, alpha_max + 1))
            
            print(f"[Numeric] Alphabet:")
            print(f"  Instance range: [{inst_min}, {inst_max}]")
            print(f"  Alphabet range: [{alpha_min}, {alpha_max}] (margin={margin})")
            print(f"  Alphabet size: {len(alphabet)}")
        else:
            # 符號型數據：使用實例中的符號 + 訓練集中的其他符號
            instance_symbols = set(test_instance)
            all_symbols = set()
            for seq in X_train:
                all_symbols.update(seq)
            
            alphabet = sorted(list(all_symbols))
            
            print(f"[Symbolic] Instance symbols: {sorted(list(instance_symbols))}")

        print(f"Alphabet size: {len(alphabet)}")
        print(f"Alphabet: {alphabet}" if len(alphabet) > 10 else f"Alphabet: {alphabet}")
        print(f"Test instance: {test_instance}")

        explainer = AnchorTabular(
            predictor=predict_fn,
            feature_names=[f'seq_{i}' for i in range(max(len(s) for s in X_train))],
            categorical_names={},
            seed=1
        )

        explainer.fit(
            automaton_type='RA',
            train_data=X_train,
            alphabet=alphabet,
            disc_perc=None
        )
        explainer.samplers[0].d_train_data = X_train

        # 4. Run Explanation
        print("\n============ Training RA Explanation ============")
        
        # Determine theory, registers, states, edit_distance based on language type
        # 關鍵改進：減小初始DRA狀態數，讓beam search從小的結構開始增長
        if lang_code.startswith('L'):
            theory_type = 'Integer'  # Integer theory 使用 EQ/NEQ，適合模式匹配
            
            # 所有 L 系列都需要至少 2 個寄存器來記住和比較符號
            num_regs = 2
            
            # 單符號重複模式 (a^n)*
            if lang_code in ['L1', 'L2A', 'L3A', 'L4A', 'L5A', 'L6A', 'L7A', 'L8A', 'L9A', 'L10A']:
                # 初始結構：小 (3 個狀態)，目標：相應大小
                # 目標狀態數 = 模式週期 + 1
                target_states = {'L1': 5, 'L2A': 5, 'L3A': 5, 'L4A': 5, 'L5A': 6, 
                                'L6A': 7, 'L7A': 8, 'L8A': 9, 'L9A': 10, 'L10A': 11}
                num_states_cfg = 4  # 初始結構：小的，包含 init + pattern + final 等基本狀態
                edit_dist = 2  # 簡單模式用較小的編輯距離
            # 多符號交替模式 (ab)^n
            elif lang_code in ['L2', 'L2AB', 'L3AB', 'L4AB', 'L5AB']:
                target_states = {'L2': 6, 'L2AB': 6, 'L3AB': 7, 'L4AB': 9, 'L5AB': 11}
                num_states_cfg = 5  # 初始結構：稍大一點，用於交替模式
                edit_dist = 3  # 交替模式需要稍大的編輯距離
            # 其他 L 系列（計數、平衡等）
            else:
                num_states_cfg = 4  # 保持初始結構小
                edit_dist = 3
        else:  # S 系列
            theory_type = 'Double'
            num_regs = 2
            # S 系列策略：從小結構開始，讓 DELTA 操作精煉
            # S12-S14 need more states due to longer alternating prefix patterns
            if lang_code in ['S12', 'S13', 'S14']:
                num_states_cfg = 6  # 從 10 減少到 6 (前綴模式需要更多狀態，但不需要初始就這麼多)
                edit_dist = 2
            elif lang_code in ['S9', 'S10', 'S11']:
                num_states_cfg = 5  # 從 8 減少到 5 (複雜邊界條件，但初始小點開始)
                edit_dist = 2
            else:
                # 簡單的 S1-S6: 初始只需要 4-5 個狀態
                num_states_cfg = 4  # 從 6 減少到 4
                edit_dist = 1  # 非常小的擾動，保持結構簡潔
        # 調整擾動距離：基於語言特性而非單個測試實例
        avg_sample_length = np.mean([len(s) for s in X_train])
        if lang_code.startswith('S'):
            # S系列：小擾動保留遞增/遞減結構
            # 基於平均長度的比例計算
            edit_dist = max(1, min(2, int(avg_sample_length // 4)))
        else:
            # L系列：中等擾動保留部分模式
            edit_dist = max(2, min(4, int(avg_sample_length // 3)))
        
        print(f"Using theory: {theory_type}, num_registers: {num_regs}, num_states: {num_states_cfg}, edit_distance: {edit_dist}")
        print(f"Average sample length: {avg_sample_length:.1f}")

        # Start timing for RA learning
        ra_start_time = time.time()
        
        # 根據語言類型設定 state_threshold - 與較小的初始DRA匹配
        # 策略：允許一些狀態增長（通過DELTA），也允許適度縮減（通過DELETE/MERGE）
        if lang_code.startswith('L'):
            # L系列：初始 4-5，允許增長到目標大小
            if lang_code in ['L2', 'L2AB', 'L3AB', 'L4AB', 'L5AB']:
                state_threshold = 8  # 允許增長到 ~8 個狀態
            else:
                state_threshold = 6  # 允許增長到 ~6 個狀態
        else:
            # S系列：初始 4-6，允許有限增長
            if lang_code in ['S12', 'S13', 'S14']:
                state_threshold = 7  # 複雜模式允許到 7 個狀態
            elif lang_code in ['S9', 'S10', 'S11']:
                state_threshold = 6  # 邊界條件允許到 6 個狀態
            else:
                state_threshold = 5  # 簡單模式允許到 5 個狀態
        
        # 準確度閾值：基於初始DRA的表現
        # 策略：首先評估初始DRA的準確度，然後設定為該值的 80-90%
        # 這樣允許beam search有改進空間，但不會卡住
        accuracy_threshold = 0.8  # 較低的閾值，讓演算法有更多探索空間
        
        explanation = explainer.explain(
            type='Tabular',
            automaton_type='RA',
            alphabet=alphabet,
            X=np.array([test_instance], dtype=object),
            edit_distance=edit_dist,
            accuracy_threshold=accuracy_threshold,
            state_threshold=state_threshold,
            delta=0.1,
            tau=0.1,
            beam_size=3,
            batch_size=500,
            min_samples_start=100,
            n_covered_ex=20,
            verbose=True, # Set to True for detailed beam search logs
            constants=[],
            theory=theory_type,
            num_states=num_states_cfg,
            num_registers=num_regs,
            output_dir=output_dir,
        )
        
        ra_learning_time = time.time() - ra_start_time
        total_time = time.time() - total_start_time

        # 5. Print Results
        print('\n============== Final Result ==============')
        print(f"Explaining instance: {test_instance}")
        print(f"Prediction: {predict_fn([test_instance])[0]}")
        
        if explanation and explanation.data:
            ra = explanation.data.get('automata', None)
            print('RA:', ra)
            print('Training Accuracy (擾動樣本):', explanation.data.get('training_accuracy', 'N/A'))
            print('Testing Accuracy (擾動樣本):', explanation.data.get('testing_accuracy', 'N/A'))
            print('Number of States:', explanation.data.get('state', 'N/A'))
            
            # 計算真正的測試集準確度
            if ra is not None:
                # 使用 DRA 的 accepts_input 方法
                ra_predictions = np.array([1 if ra.accepts_input(list(seq), debug=False) else 0 for seq in X_test])
                true_test_acc = np.mean(ra_predictions == y_test)
                print(f'True Test Accuracy (X_test): {true_test_acc:.3f}')
                
                # 分析錯誤
                false_positive = np.sum((ra_predictions == 1) & (y_test == 0))
                false_negative = np.sum((ra_predictions == 0) & (y_test == 1))
                print(f'  - False Positives: {false_positive}')
                print(f'  - False Negatives: {false_negative}')
        else:
            print("Explanation failed or did not produce a valid automaton.")
        
        # Print timing information
        print(f"\n============== Timing ==============")
        print(f"Data generation time: {data_gen_time:.2f}s")
        print(f"RA learning time: {ra_learning_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")

    # Restore stdout
    sys.stdout = original_stdout
    print(f"Results for {lang_code} saved to: {output_dir}")

print(f"\n{'='*60}")
print(f"All benchmark results saved to: test_result/benchmark_*/")
print(f"{'='*60}")
