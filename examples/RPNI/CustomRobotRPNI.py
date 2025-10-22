import sys, os
sys.path.insert(0, './src')
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA')))

from tee import Tee
import numpy as np
import pickle
from modified_modules.alibi.explainers import AnchorTabular

from dataset_loader import fetch_custom_dataset  # 通用載入器
from robot_operation import robot_instance
from automaton.dfa_operatopn import get_base_dfa, simplify_dfa, dfa_intersection_any, merge_linear_edges, merge_parallel_edges
from automaton.utils import dfa_to_mata, explain_axp_cxp, get_test_word
from automaton.learner import AUTO_INSTANCE
from language.explain import Language

# 載入資料
b = fetch_custom_dataset(
    source="datasets/robot.csv",
    mode="tabular",
    target_col="label",
    return_X_y=False
)
data = b.data
raw_data = b.raw_data
target = b.target
feature_names = b.feature_names # ['s0','s1',...]
category_map = b.category_map # {col_idx: [class_names...]}

# 預測模型
def robot_predict_fn(X):
    X = np.asarray(X)
    X = np.atleast_2d(X).astype(int) # 如果輸入是整數索引，先轉型

    preds = []
    for row in X:
        # 用 category_map 逐欄把編碼值 → 類別字串
        seq = [
            category_map[j][int(v)] if j in category_map and 0 <= int(v) < len(category_map[j])
            else str(int(v))
            for j, v in enumerate(row)
        ]
        preds.append(int(robot_instance.is_valid_path(seq)))
    return np.array(preds, dtype=int)

# fit anchor explainer
explainer = AnchorTabular(
    predictor=robot_predict_fn,
    feature_names=feature_names,
    categorical_names=dict(category_map),
    seed=1
)
explainer.fit(data)
explainer.samplers[0].d_train_data = data # 設定 X 為原始訓練資料

# 解釋句子
test_instance = data[1]
raw_test_instance = raw_data[1]

# explainer 參數
learn_type = 'Tabular'
coverage_samples = 1000
batch_size = 50
n_covered_ex = 1000
min_samples_start = 30
threshold = 0.95

with open("TestRobot.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) # 同時輸出到終端和檔案
    print("\nInstance: %s" % raw_test_instance)
    print("Prediction: %s" % robot_predict_fn([test_instance]))
    print("\n============Training step============")

    explanation = explainer.explain(
        learn_type,
        test_instance,
        coverage_samples=coverage_samples,
        batch_size=batch_size,
        n_covered_ex=n_covered_ex,
        min_samples_start=min_samples_start,
        threshold=threshold,
        beam_size=1
    )

    # Anchor 結果
    print('\n==============Result==============')
    print('Anchor:', explanation.anchor)
    print('Raw anchor: %s' % (explanation.raw['feature']))
    mab = explainer.mab # 取出學習紀錄
    print(f"Feature values:{mab.sample_fcn.feature_values}\n")

     # 生成自動機
    origin_dfa, testing_data = AUTO_INSTANCE.create_automata(mab.type, tuple(explanation.raw['feature']), mab.state)
    print(f"Origin DFA:{origin_dfa}")

    # 找 AXp、CXp
    # test_word = [int(v) for v in data[1]]
    # print("test_word:", test_word)
    # state_map, alphabet_map = dfa_to_mata(mab.dfa, "dfa_explicit.mata")
    # test_word = get_test_word("dfa_explicit.mata", alphabet_map) # 取得測試路徑
    # explanation_engine = Language()
    # result = explanation_engine.explain_word(
    #     "dfa_explicit.mata",
    #     from_mata=True,
    #     word=[1, 1, 2, 2, 1],
    #     ascii=False,
    #     target_axp=True,
    #     bootstrap_cxp_size_1=False,
    #     print_exp=True
    # )
    # explain_axp_cxp(result["axps"], result["cxps"], alphabet_map) # 轉換axps、cxps成文字
    
    # 計算 DFA Intersection
    features = explanation.raw['feature'] # 取出 anchor 值
    alphabet_map = {} # 建立 dfa 的字母表映射
    for i in mab.sample_fcn.feature_values:
        if i not in alphabet_map:
            alphabet_map[i] = []
        for j in range(len(mab.sample_fcn.feature_values[i])):
            alphabet_map[i].append(j)
    sub_dfa = get_base_dfa(learn_type, alphabet_map, features, test_instance)
    inter_dfa = dfa_intersection_any(origin_dfa, sub_dfa) # 交集

    # pickle 存 dfa、測試資料 (不加位置 機器人)
    # result = []
    # for idx in range(len(mab.testing_data)):
    #     data = list(mab.testing_data[idx])
    #     result.append((data, mab.state['coverage_label'][idx]))

    # with open("robot_dfa.pkl", "wb") as f:
    #     pickle.dump(inter_dfa, f)
    # with open("robot_testing_data.pkl", "wb") as f:
    #     pickle.dump(result, f)

    # 化簡 DFA
    inter_dfa.make_input_complete()
    inter_dfa.minimize()
    # print("intersection dfa:", inter_dfa)
    final_dfa = simplify_dfa(inter_dfa, learn_type) # 合併邊
    print(f"Final DFA:{final_dfa}\n")

    sys.stdout = sys.__stdout__ # 恢復 stdout