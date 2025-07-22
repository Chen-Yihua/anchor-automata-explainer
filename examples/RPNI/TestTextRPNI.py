import sys
sys.path.insert(0, './src')
sys.path.insert(0, './modified_packages')
from tee import Tee
import spacy
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from alibi.explainers import AnchorText
from alibi.datasets import fetch_movie_sentiment
from alibi.utils import spacy_model
from dfa_operatopn import dfa_intersection, get_base_dfa, merge_linear_edges, merge_parallel_edges

np.random.seed(0)

# 載入資料
movies = fetch_movie_sentiment()
data = movies.data
labels = movies.target
target_names = movies.target_names
class_names = movies.target_names

# 切分資料
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

# 訓練模型
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
clf = LogisticRegression(solver='liblinear')
clf.fit(vectorizer.transform(train), train_labels)

# 進行預測
predict_fn = lambda x: clf.predict(vectorizer.transform(x))
preds_train = predict_fn(train)
preds_val = predict_fn(val)
preds_test = predict_fn(test)
print('Train accuracy: %.3f' % accuracy_score(train_labels, preds_train))
print('Validation accuracy: %.3f' % accuracy_score(val_labels, preds_val))
print('Test accuracy: %.3f' % accuracy_score(test_labels, preds_test))

# 載入 spaCy model (explainers 需要進行斷詞和部分前處理)
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
text = data[10] # instance to be explained
text_length = len(text.split())
pred = class_names[predict_fn([text])[0]] # compute class prediction

coverage_samples = 1000
batch_size = 50
min_samples_start = 30
n_covered_ex = 1000
threshold = 0.95

with open("TestTextRPNI.txt", "w", encoding="utf-8") as log_file:
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
    alphabet_map = {} # 建立 dfa 的 alphabet 映射表
    for i in range(text_length):
        alphabet_map[i] = [0.0, 1.0]

    sub_dfa = get_base_dfa(alphabet_map) # 子 dfa
    print("sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection(mab.dfa, sub_dfa)
    print("intersection dfa:", inter_dfa)
    dfa = merge_parallel_edges(inter_dfa)
    dfa = merge_linear_edges(dfa)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout