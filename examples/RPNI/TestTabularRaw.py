# 使用原始樣本 (["age =49", "education = Bachelors",...]) 製作自動機
# 若需要此例子，需先修改被動樣本格式轉換方法

import sys
sys.path.insert(0, './src')
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from alibi.explainers import AnchorTabular
from anchor_dfa_learning import AUTO_INSTANCE
from alibi.datasets import fetch_adult

class BankWithTabular:
    def __init__(self):
        self.prediction = None
        self.learned_positives = []
        self.learned_negatives = []

    
    def run(self):
        adult = fetch_adult()
        adult.keys()

        data = adult.data
        target = adult.target
        feature_names = adult.feature_names
        category_map = adult.category_map

        np.random.seed(0)
        data_perm = np.random.permutation(np.c_[data, target])
        data = data_perm[:,:-1]
        target = data_perm[:,-1]

        idx = 30000
        X_train,Y_train = data[:idx,:], target[:idx]
        X_test, Y_test = data[idx+1:,:], target[idx+1:]

        # Ordinal features
        ordinal_features = [x for x in range(len(feature_names)) if x not in list(category_map.keys())]
        ordinal_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                            ('scaler', StandardScaler())])

        # Categorical features
        categorical_features = list(category_map.keys())
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore'))])


        preprocessor = ColumnTransformer(transformers=[('num', ordinal_transformer, ordinal_features),
                                                    ('cat', categorical_transformer, categorical_features)])
        preprocessor.fit(X_train)

        #　Train Random Forest model
        np.random.seed(0)
        clf = RandomForestClassifier(n_estimators=50, random_state=0)
        clf.fit(preprocessor.transform(X_train), Y_train)

        predict_fn = lambda x: clf.predict(preprocessor.transform(x))
        print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
        print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))

        #　fit anchor explainer
        explainer = AnchorTabular(predict_fn, feature_names, categorical_names=category_map, seed=1)
        explainer.fit(X_train, disc_perc=[25, 50, 75])
        explainer.samplers[0].d_train_data = X_train # 設定 d_train_data 為原始訓練資料

        # 解釋
        idx = 2 # 2nd row of the test set
        coverage_samples = 1000
        batch_size = 50
        n_covered_ex = 1000
        min_samples_start = 30
        threshold = 0.75
        explanation = explainer.explain('Tabular', X_test[idx], coverage_samples=coverage_samples, batch_size=batch_size, n_covered_ex=n_covered_ex, min_samples_start=min_samples_start, threshold=threshold, beam_size=1)
        
        # 印出 anchor 結果
        print('Anchor: %s' % (' AND '.join(explanation.anchor)))
        print('Anchor Precision: %.4f' % explanation.precision)
        print('Anchor Coverage: %.4f' % explanation.coverage)
        mab = explainer.mab # 取出學習紀錄
        sampler = explainer.samplers[0] # 取出 sampler
        
        # 取出被動學習樣本
        final_ex = explanation.data['raw']['examples'][-1]
        positives_samples = final_ex['covered_true']   # 符合 anchor 且跟原始預測相同的原始樣本
        negatives_samples = final_ex['covered_false']  # 符合 anchor 但預測不同的原始樣本   
        raw_pos_samples = [self.decode_to_raw_with_bin(sampler, sample) for sample in positives_samples] # 把 dis_samples 轉成原始樣本 (with bin)
        raw_neg_samples = [self.decode_to_raw_with_bin(sampler, sample) for sample in negatives_samples]
        passive_data = AUTO_INSTANCE.convert_to_alergia_format(raw_pos_samples, raw_neg_samples) # 轉成 RPNI 要的格式
        print('被動學習樣本數量: %d' % len(passive_data))

        # 學 DFA
        dfa = AUTO_INSTANCE.learn_dfa(passive_data)

        # 計算 DFA Precision
        raw_pos_samples = AUTO_INSTANCE.convert_to_alergia_format(raw_pos_samples) # 轉成 RPNI 要的格式
        pos_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in raw_pos_samples) # 計算 DFA 機率
        passive_data_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in passive_data)
        dfa_precision = pos_prob / passive_data_prob if len(passive_data) > 0 else 0
        print(f"MC Precision: {dfa_precision:.4f}")

        # 計算 DFA Coverage
        anchor_raw = explanation.data['raw']['feature'] # 取出 anchor 的原始索引
        binary_anchor_raw = [] # 取得 anchor 的二元編碼
        for bin_idx, raw_idx in sampler.enc2feat_idx.items():
            if raw_idx in anchor_raw:
                binary_anchor_raw.append(bin_idx)
        covered_idx = sorted(mab.state['t_coverage_idx'][tuple(binary_anchor_raw)]) # 取得符合 anchor 條件的覆蓋樣本原始索引

        covered_samples = [ # 取出符合 anchor 條件的樣本 (接受的樣本)
            explainer.mab.state['coverage_raw'][i]
            for i in covered_idx
        ]
        anchor_covered_samples = [ # 將符合 anchor 的二元 coverage_data 轉成原始樣本(分 bin)
            self.decode_to_raw_with_bin(sampler, samples)
            for samples in covered_samples
        ]
        anchor_covered_samples = AUTO_INSTANCE.convert_to_alergia_format(anchor_covered_samples, None) # 分子樣本

        all_covered_samples = [ # 將二元 coverage_data 轉成原始樣本(分 bin)
            self.decode_to_raw_with_bin(sampler, samples)
            for samples in explainer.mab.state['coverage_raw']
        ]
        all_covered_samples = AUTO_INSTANCE.convert_to_alergia_format(all_covered_samples, None) # 分母樣本
        
        accepted_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in anchor_covered_samples) 
        coverage_data_prob = sum(AUTO_INSTANCE.calculate_path_probability(sample) for sample in all_covered_samples)
        dfa_coverage = accepted_prob / coverage_data_prob if coverage_samples > 0 else 0
        print(f"accepted prob: {accepted_prob:.4f}")
        print(f"MC Coverage: {dfa_coverage:.4f}")


instance = BankWithTabular()
instance.run()
print("==================================")
