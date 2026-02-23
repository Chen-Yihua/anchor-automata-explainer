import sys
sys.path.insert(0, './src')
from tee import Tee
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from alibi.explainers import AnchorTabular
from alibi.datasets import fetch_adult
from automaton.dfa_operation import get_base_dfa, simplify_dfa, dfa_intersection_any

np.random.seed(0)

# 載入資料
adult = fetch_adult()
adult.keys()
data = adult.data
target = adult.target
feature_names = adult.feature_names
category_map = adult.category_map

# 切分資料
data_perm = np.random.permutation(np.c_[data, target])
data = data_perm[:,:-1]
target = data_perm[:,-1]

idx = 30000
X_train,Y_train = data[:idx,:], target[:idx]
X_test, Y_test = data[idx+1:,:], target[idx+1:]

# 轉換 Ordinal features
ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]

# 分 bin 或是二元化數值資料
df = pd.DataFrame(X_train, columns=feature_names)
bins_dict = {}
bin_names = {}
for i in ordinal_features:
    col = feature_names[i]
    unique_vals = df[col].nunique()

    if unique_vals < 4: # 若值太少，則跳過
        continue
    else: # 若偏態（90% 是 0），做成是否 > 0 的二元特徵
        if (df[col] == 0).mean() > 0.9:
            df[col + "_pos"] = (df[col] > 0).astype(int)
            continue
        else: # 用 qcut 分成 4 bin
            bins, bins_edges = pd.qcut(df[col], q=[0, 0.25, 0.5, 0.75, 1.0], retbins=True, duplicates='drop', labels=False)
            df[col + "_bin"] = bins
            bins_dict[i] = bins_edges
            bin_names[col] = [f"{col}_bin_{j}" for j in range(len(bins_edges)-1)]

# 合併處理完的數值資料到訓練資料
for i in bins_dict: # 分 bin 後的數值資料
    col = feature_names[i]
    X_train[:, i] = pd.cut(X_train[:, i], bins=bins_dict[i], labels=False, include_lowest=True)
    X_test[:, i] = pd.cut(X_test[:, i], bins=bins_dict[i], labels=False, include_lowest=True)
for i in ordinal_features: # 二元化後的數值資料
    col = feature_names[i]
    if col + "_pos" in df.columns:
        X_train[:, i] = df[col + "_pos"].values
        df_test = pd.DataFrame(X_test, columns=feature_names)
        X_test[:, i] = (df_test[col] > 0).astype(int).values

# 補齊 NaN
X_train = np.array(X_train)
X_test = np.array(X_test)
X_train = np.where(np.isnan(X_train), -1, X_train)
X_test = np.where(np.isnan(X_test), -1, X_test)

# 轉換 Categorical features
categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], 
    remainder='passthrough' # 把 bin 後的 ordinal 欄位 passthrough
)
preprocessor.fit(X_train)

#　訓練模型
clf = RandomForestClassifier(n_estimators=50, random_state=0)
clf.fit(preprocessor.transform(X_train), Y_train)
predict_fn = lambda x: clf.predict(preprocessor.transform(x))
print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))

# fit anchor explainer
categorical_names = {}
categorical_names.update(category_map) # category_map: categorical col → category names
for i in ordinal_features:
    col = feature_names[i]
    if i in bins_dict: # 分 bin 的數值特徵 
        categorical_names[i] = [str(k) for k in range(len(bins_dict[i]) - 1)]
    elif col + "_pos" in df.columns: # 二值化的數值特徵
        categorical_names[i] = [0, 1]

explainer = AnchorTabular(
    predict_fn, 
    feature_names, 
    categorical_names=categorical_names, 
    seed=1
)
explainer.fit(X_train, disc_perc=[25, 50, 75])
explainer.samplers[0].d_train_data = X_train # 設定 d_train_data 為原始訓練資料

# 解釋 tabular
test_instance = X_test[2]

# explainer 參數
learn_type = 'Tabular'
coverage_samples = 1000
batch_size = 50
n_covered_ex = 1000
min_samples_start = 30
threshold = 0.95

with open("TestTabularRPNI.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file)

    print("Tabular: %s" % test_instance)
    print("Prediction: < 50 k")
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
    print('Anchor: %s' % (' AND '.join(explanation.anchor)))
    mab = explainer.mab # 取出學習紀錄

    # 計算 DFA Intersection
    alphabet_map = {} # 建立 dfa 的字母表映射
    features = explanation.raw['feature'] # anchor 值

    for i in mab.sample_fcn.feature_values:
        if i not in alphabet_map:
            alphabet_map[i] = []
        for j in range(len(mab.sample_fcn.feature_values[i])):
            alphabet_map[i].append(j)

    sub_dfa = get_base_dfa(learn_type, alphabet_map, features, test_instance)  # 子 dfa
    print("sub dfa:", sub_dfa)

    inter_dfa = dfa_intersection_any(mab.dfa, sub_dfa) # 交集
    inter_dfa.make_input_complete()
    inter_dfa.minimize()
    print("intersection dfa:", inter_dfa)

    dfa = simplify_dfa(inter_dfa, learn_type)
    print("final dfa:", dfa)

    sys.stdout = sys.__stdout__ # 恢復 stdout