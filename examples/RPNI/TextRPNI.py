import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA')))
sys.path.insert(0, os.path.abspath('./src'))
sys.path.insert(0, os.path.abspath('.'))

from tee import Tee
from spacy.lang.en import English
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from modified_modules.alibi.explainers import AnchorText
from data.dataset_loader import fetch_custom_dataset
from learner.dfa_learner import get_base_dfa, simplify_dfa, dfa_intersection_any

np.random.seed(0)

# 載入資料
MOVIESENTIMENT_URLS = ['https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz',
                       'http://www.cs.cornell.edu/People/pabo/movie-review-data/rt-polaritydata.tar.gz']

url = MOVIESENTIMENT_URLS[0]
movies = fetch_custom_dataset(
    source=url,
    mode="text",
    return_X_y=False
)
data = movies.data
labels = movies.target
target_names = movies.target_names
class_names = movies.target_names

# 預測模型
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 3),   
    max_features=20000,  
)
vectorizer.fit(train)

X_train = vectorizer.transform(train)
X_val = vectorizer.transform(val)
X_test = vectorizer.transform(test)

from sklearn.svm import LinearSVC
clf = LinearSVC(
    C=1.0,
    max_iter=2000,
    random_state=42,
    dual=False,
    class_weight='balanced'
)
clf.fit(X_train, train_labels)

predict_fn = lambda x: clf.predict(vectorizer.transform(x))

preds_train = predict_fn(train)
preds_val = predict_fn(val)
preds_test = predict_fn(test)

print("="*60)
train_acc = accuracy_score(train_labels, preds_train)
val_acc = accuracy_score(val_labels, preds_val)
test_acc = accuracy_score(test_labels, preds_test)

print(f'Train accuracy: {train_acc:.4f}')
print(f'Validation accuracy: {val_acc:.4f}')
print(f'Test accuracy: {test_acc:.4f}')
print("="*60)

# sampler 參數
nlp = English()

sampling_strategy = 'unknown' # unknown、similarity、language_model、shap
sample_proba = 0.5
use_proba = True
top_n = 3
temperature = 0.1
language_model = None
seed = 0
automaton_type = 'DFA'

# fit anchor explainer
explainer = AnchorText(
    predictor=predict_fn,
    sampling_strategy=sampling_strategy,
    sample_proba=sample_proba,
    nlp=nlp,
    # clf=clf, # shap
    # vectorizer=vectorizer, # shap
    use_proba=use_proba,
    top_n=top_n,
    temperature=temperature,
    # language_model=language_model,
    seed=seed
)

# 解釋句子
test_instance = 'barely fun and ordinary .' # interesting , but not compelling .
text_length = len(test_instance.split())
pred = class_names[predict_fn([test_instance])[0]] 

# explainer 參數
learn_type = 'Text'
threshold = 0.98
delta = 0.1
tau = 0.15
batch_size = 100
coverage_samples = 1000
beam_size = 1
max_anchor_size = None
min_samples_start = 100
n_covered_ex = 20
edit_distance = 4


with open("TestText.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) 
    print("\nText: %s" % test_instance)
    print("Prediction: %s" % pred)
    print("\n============Training step============")
    
    explanation = explainer.explain(
        learn_type, 
        automaton_type=automaton_type,
        text=test_instance, 
        edit_distance=edit_distance,
        delta = delta,
        tau = tau, 
        beam_size = beam_size, 
        max_anchor_size = max_anchor_size,
        coverage_samples=coverage_samples, 
        batch_size=batch_size, 
        min_samples_start=min_samples_start, 
        n_covered_ex=n_covered_ex, 
        threshold=threshold,
        verbose=True
    )

    # # Anchor 結果
    print('\n==============Result==============')
    print('DFA:', explanation.dfa)
    print('Training Accuracy:', explanation.training_accuracy)
    print('Testing Accuracy:', explanation.testing_accuracy)
    print('State:', explanation.state)
    
    mab = explainer.mab # 學習紀錄

    # 生成自動機
    # origin_dfa, testing_data = AUTO_INSTANCE.create_automata(mab.type)
    # print(f"Origin DFA:{origin_dfa}\n")
    # 找 AXp、CXp
    # state_map, alphabet_map = dfa_to_mata(mab.dfa, "dfa_explicit.mata")
    # test_word = get_test_word("dfa_explicit.mata", alphabet_map) # 取得測試路徑
    # explanation_engine = Language()
    # result = explanation_engine.explain_word(
    #     "dfa_explicit.mata",
    #     from_mata=True,
    #     word=list(test_word),
    #     ascii=False,
    #     target_axp=True,
    #     bootstrap_cxp_size_1=False,
    #     print_exp=True
    # )
    # explain_axp_cxp(result["axps"], result["cxps"], alphabet_map) # 轉換axps、cxps成文字
    
    # pickle 存測試資料
    # with open("text_testing_data.pkl", "wb") as f:
    #     pickle.dump(mab.state["coverage_data"], f)

    # 使用固定測試集，重新跑 evaluation
    # import pickle
    # with open("text_testing_data.pkl", "rb") as f:
    #     result = pickle.load(f)
    # positive_samples = [sample for sample, label in result if label == 1]
    # negative_samples = [sample for sample, label in result if label == 0]
    # print("new testing data : ")
    # AUTO_INSTANCE.get_evaluation(learn_type,
    #     tuple(sorted(explanation.data['raw']['feature'])),
    #     mab.state,
    #     mab.dfa,
    #     positive_samples,
    #     negative_samples
    # )

    # 計算 DFA Intersection
    # Extract alphabet_map from the test instance words with correct position encoding
    words = test_instance.split()
    alphabet_map = {}
    for pos, word in enumerate(words):
        # Use the same format as RPNI learning: p{pos}_(word)
        alphabet_map[pos] = [f'p{pos}_({word})', f'p{pos}_(UNK)']
    
    sub_dfa = get_base_dfa(learn_type, alphabet_map)
    # print("Sub dfa:", sub_dfa)
    inter_dfa = dfa_intersection_any(explanation.dfa, sub_dfa)
    print("Inter dfa:", inter_dfa)
    final_dfa = simplify_dfa(inter_dfa, learn_type) # 合併邊
    print("Final dfa:", final_dfa)

    
    # pickle 存 dfa、測試資料
    # result = []
    # for idx in range(len(mab.testing_data)):
    #     result.append((mab.testing_data[idx], mab.state['coverage_label'][idx]))
    # with open("text_dfa.pkl", "wb") as f:
        # pickle.dump(inter_dfa, f)
    # with open("text_testing_data.pkl", "wb") as f:
        # pickle.dump(result, f)

    # 化簡 DFA
    # inter_dfa.make_input_complete()
    # inter_dfa.minimize()
    # print("intersection dfa:", inter_dfa)
    # final_dfa = simplify_dfa(inter_dfa, learn_type) # 合併邊
    # print(f"Final DFA:{final_dfa}\n")
    
    sys.stdout = sys.__stdout__