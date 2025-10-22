import sys
sys.path.insert(0, './src')
from tee import Tee
import numpy as np
from alibi.explainers import AnchorTabular
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from robot_operation import robot_instance
from automaton.dfa_operatopn import dfa_intersection, get_base_dfa, merge_linear_edges, merge_parallel_edges

def robot_predict_fn(X):
    color_map = {0: 'green', 1: 'yellow', 2: 'blue'}
    X_int = X.astype(int)
    X_str = np.vectorize(color_map.get)(X_int)
    result = np.array([int(robot_instance.is_valid_path(row)) for row in X_str])
    return result

# 載入資料
robot = robot_instance.fetch_robot()
robot.keys()
data = robot.data
target = robot.target
feature_names = robot.feature_names
category_map = robot.category_map
raw_data = robot.raw_data

# 轉換 Categorical features
categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], 
    remainder='passthrough' # 把 bin 後的 ordinal 欄位 passthrough
)
preprocessor.fit(data)

# fit anchor explainer
categorical_names = {}
categorical_names.update(category_map) # category_map: categorical col → category names

explainer = AnchorTabular(
    predictor=robot_predict_fn,
    feature_names=feature_names,
    categorical_names=categorical_names,
    seed=1
)
explainer.fit(data)
explainer.samplers[0].d_train_data = data # 設定 X 為原始訓練資料

# 解釋句子
test_instance = data[2]
raw_test_instance = raw_data[2]
coverage_samples = 1000
batch_size = 50
n_covered_ex = 1000
min_samples_start = 30
threshold = 0.95

with open("TestRobotRPNI.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) # 同時輸出到終端和檔案
    print('解釋實例:', raw_test_instance, '預測:', robot_predict_fn(test_instance))

    explanation = explainer.explain(
        'Tabular',
        test_instance,
        coverage_samples=coverage_samples,
        batch_size=batch_size,
        n_covered_ex=n_covered_ex,
        min_samples_start=min_samples_start,
        threshold=threshold,
        beam_size=1
    )

    print('Anchor:', explanation.anchor)

    mab = explainer.mab # 取出學習紀錄

    # 計算 DFA Intersection
    alphabet_map = {} # 建立 dfa 的字母表映射
    for i in mab.sample_fcn.feature_values:
        if i not in alphabet_map:
            alphabet_map[i] = []
        for j in range(len(mab.sample_fcn.feature_values[i])):
            alphabet_map[i].append(j)

    sub_dfa = get_base_dfa(alphabet_map) # 子 dfa
    print("sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection(mab.dfa, sub_dfa)
    print("intersection dfa:", inter_dfa)
    dfa = merge_parallel_edges(inter_dfa)
    dfa = merge_linear_edges(dfa)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout