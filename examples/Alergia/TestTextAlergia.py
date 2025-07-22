import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # surpressing some transformers' output

import spacy
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from alibi.explainers import AnchorText
from alibi.datasets import fetch_movie_sentiment
from alibi.utils import spacy_model
# from alibi.utils import DistilbertBaseUncased, BertBaseUncased, RobertaBase
from anchor_dfa_learning import AUTO_INSTANCE, get_alergia_samples, get_precision_samples, add_position_to_sample, get_pos_samples, get_neg_samples, get_covered_samples

# 載入資料
movies = fetch_movie_sentiment() # movies.keys(): dict_keys(['data', 'target', 'target_names'])
data = movies.data
labels = movies.target
target_names = movies.target_names

train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

# fit 模型
vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)

np.random.seed(0)
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

# 載入 spaCy model
model = 'en_core_web_md'
spacy_model(model=model)
nlp = spacy.load(model)


# 解釋句子
class_names = movies.target_names
text = data[0] # instance to be explained
print("* Text: %s" % text)
pred = class_names[predict_fn([text])[0]] # compute class prediction
alternative =  class_names[1 - predict_fn([text])[0]]
print("* Prediction: %s" % pred)

explainer = AnchorText(
    predictor=predict_fn,
    sampling_strategy='unknown',
    nlp=nlp,
)
coverage_samples = 1000
batch_size = 50
min_samples_start = 30
n_covered_ex = 1000
threshold = 0.95
explanation = explainer.explain(text, coverage_samples=coverage_samples, batch_size=batch_size, min_samples_start=min_samples_start, n_covered_ex=n_covered_ex, threshold=threshold)

# Anchor 結果
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Anchor Precision: %.2f' % explanation.precision)
print('Anchor Coverage: %.4f' % explanation.coverage)
sampler = explainer.sampler # 取出 sampler
mab = explainer.mab # 取出學習紀錄

# 取出被動學習樣本
passive_data = get_alergia_samples('Text', mab)
passive_data = add_position_to_sample(passive_data)
passive_data = AUTO_INSTANCE.convert_to_alergia_format(passive_data)
print('被動學習樣本數量: %d' % len(passive_data))

# 生成 dfa
dfa = AUTO_INSTANCE.learn_dfa(list(passive_data))

# 計算 DFA Precision
pos_samples = get_pos_samples('Text', explanation, mab) # 取得符合 anchor 條件的正例
pos_samples = add_position_to_sample(pos_samples)
pos_samples = AUTO_INSTANCE.convert_to_alergia_format(pos_samples)

precision_samples = get_precision_samples('Text', explanation, sampler, mab) # 取得符合 anchor 條件的 precision 樣本
precision_samples = add_position_to_sample(precision_samples)
precision_samples = AUTO_INSTANCE.convert_to_alergia_format(precision_samples)

pos_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in pos_samples) # 計算 DFA 機率
passive_data_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in passive_data)
dfa_precision = pos_prob / passive_data_prob if len(precision_samples) > 0 else 0
print(f"Automaton Precision: {dfa_precision:.4f}")

# 計算 DFA Coverage
all_covered_samples = mab.state['coverage_data']
all_covered_samples = add_position_to_sample(all_covered_samples)
all_covered_samples = AUTO_INSTANCE.convert_to_alergia_format(all_covered_samples)

anchor_covered_samples = get_covered_samples('Text', explanation, mab)
anchor_covered_samples = add_position_to_sample(anchor_covered_samples)
anchor_covered_samples = AUTO_INSTANCE.convert_to_alergia_format(anchor_covered_samples)

accepted_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in anchor_covered_samples) 
coverage_data_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in all_covered_samples)
dfa_coverage = accepted_prob / coverage_data_prob if coverage_samples > 0 else 0
print(f"Automaton Coverage: {dfa_coverage:.4f}")