import sys
sys.path.insert(0, './src')
from tee import Tee
import spacy
from alibi.explainers import AnchorText
from alibi.utils import spacy_model
from robot_operation import robot_instance
from automaton.dfa_operatopn import dfa_intersection, get_base_dfa, merge_parallel_edges, merge_linear_edges

# 進行預測
predict_fn = lambda x: robot_instance.robot_predict_fn(x)

# 載入 spaCy model
model = 'en_core_web_md'
spacy_model(model=model)
nlp = spacy.load(model)

# fit anchor explainer
explainer = AnchorText(
    predictor=predict_fn,
    sampling_strategy='unknown',
    nlp=nlp,
)

# 解釋句子
text = "yellow yellow yellow" # instance to be explained
text_length = len(text.split())
pred = predict_fn([text])[0] # compute class prediction

coverage_samples = 1000
batch_size = 50
min_samples_start = 30
n_covered_ex = 1000
threshold = 0.95

with open("TestRobotTextRPNI.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) # 同時輸出到終端和檔案
    print("Text: %s" % text)
    print("Prediction: %s" % pred)

    explanation = explainer.explain('Text', text, coverage_samples=coverage_samples, batch_size=batch_size, min_samples_start=min_samples_start, n_covered_ex=n_covered_ex, threshold=threshold)

    # Anchor 結果
    print('Anchor: %s' % (' AND '.join(explanation.anchor)))
    # print('Anchor Precision: %.2f' % explanation.precision)
    # print('Anchor Coverage: %.4f' % explanation.coverage)
    mab = explainer.mab # 取出學習紀錄

    # 計算 DFA Intersection
    alphabet_map = {} # 建立 dfa 的字母表映射
    for i in range(text_length):
        alphabet_map[i] = [0.0, 1.0]

    sub_dfa = get_base_dfa(alphabet_map)
    print("sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection(mab.dfa, sub_dfa)
    print("intersection dfa:", inter_dfa)
    dfa = merge_parallel_edges(inter_dfa)
    dfa = merge_linear_edges(dfa)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout