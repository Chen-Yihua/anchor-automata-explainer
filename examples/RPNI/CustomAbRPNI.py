import re
import sys
sys.path.insert(0, './src')
from tee import Tee
import numpy as np
from alibi.explainers import AnchorTabular
from automaton.dfa_operation import dfa_intersection, get_base_dfa, merge_linear_edges, merge_parallel_edges
from data.dataset_loader import fetch_custom_dataset  # 通用載入器

# 載入資料
b = fetch_custom_dataset(
    source="datasets/ab_tabular.csv",
    mode="tabular",
    target_col="label",
    return_X_y=False
)
data = b.data
raw_data = b.raw_data
target = b.target
feature_names = b.feature_names
category_map = b.category_map 

def decode_row(row, cmap):
    out = []
    for j, v in enumerate(row):
        out.append(cmap[j][int(v)] if j in cmap else str(v))
    return out
raw_data = np.array([decode_row(r, category_map) for r in data], dtype=object)

# 預測模型
PATTERN = re.compile(r"^(?:ab){3}$")

def check_pattern(s):
    if isinstance(s, (list, tuple, np.ndarray)): # 有可能是 list[str] 或 str
        s = "".join(map(str, s))
    else:
        s = str(s)
    s = "".join(ch for ch in s if ch in {"a", "b"})
    s = s.replace(" ", "") # 移除空白
    return bool(PATTERN.fullmatch(s))

def predict_fn(X):
    X = np.asarray(X)
    X = np.atleast_2d(X).astype(int)
    preds = []
    for row in X:
        seq = [
            category_map[j][int(v)] if j in category_map and 0 <= int(v) < len(category_map[j])
            else str(int(v))
            for j, v in enumerate(row)
        ]
        preds.append(int(check_pattern(seq)))
    return np.array(preds, dtype=int)

# fit anchor explainer
explainer = AnchorTabular(
    predictor=predict_fn,
    feature_names=feature_names,
    categorical_names=dict(category_map),
    seed=1
)
explainer.fit(data)

# 解釋句子
test_instance = data[21]
raw_test_instance = raw_data[21]
text_length = len(test_instance)
coverage_samples = 1000
batch_size = 50
min_samples_start = 30
n_covered_ex = 1000
threshold = 0.95

with open("CustomabRPNI.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) # 同時輸出到終端和檔案
    print('解釋實例:', raw_test_instance, '預測:', predict_fn([test_instance]))

    explanation = explainer.explain(
        'Tabular', 
        test_instance, 
        coverage_samples=coverage_samples, 
        batch_size=batch_size, 
        min_samples_start=min_samples_start, 
        n_covered_ex=n_covered_ex, 
        threshold=threshold
    )

    # Anchor 結果
    print('Anchor: %s' % (' AND '.join(explanation.anchor)))
    mab = explainer.mab # 取出學習紀錄

    # 計算 DFA Intersection
    alphabet_map = {} # 建立 dfa 的 alphabet 映射表
    for i in mab.sample_fcn.feature_values:
        if i not in alphabet_map:
            alphabet_map[i] = []
        for j in range(len(mab.sample_fcn.feature_values[i])):
            alphabet_map[i].append(j)

    sub_dfa = get_base_dfa(alphabet_map)
    # print("sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection(mab.dfa, sub_dfa)
    # print("intersection dfa:", inter_dfa)
    dfa = merge_parallel_edges(inter_dfa)
    dfa = merge_linear_edges(dfa)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout