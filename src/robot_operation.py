import numpy as np
from typing import Optional, Tuple, Union
from modified_modules.alibi.utils.data import Bunch

class RobotPredictor:
    def __init__(self, alphabets=None):
        self.alphabet = alphabets if alphabets else ['yellow', 'blue', 'green']
        # self.learned_positives = []
        # self.learned_negatives = []
        self.unpredictable_count  = 0  # 用於計算無法預測 (預測結果非唯一) 的樣本數量
        self.coverage_data = None  # 用於存儲 coverage data (初始抽樣樣本)

    def fetch_robot(self, features_drop: Optional[list] = None, return_X_y: bool = False, url_id: int = 0) -> \
        Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
        """
        產生 robot tabular
        """
        import pandas as pd
        from itertools import product
        from sklearn.preprocessing import LabelEncoder

        # download data
        colors = ['green', 'yellow', 'blue'] # 顏色
        path_length = 3 # 路徑長度
        feature_names = [f"step_{i+1}" for i in range(path_length)] # feature 名稱
        category_map = {i: colors for i in range(path_length)}

        raw_data = np.array(list(product(colors, repeat=path_length)), dtype=object) # 產生所有組合
        df = pd.DataFrame(raw_data, columns=feature_names) # 轉成 DataFrame
        df['label'] = df.apply(lambda row: self.is_valid_path(row.values), axis=1) # 增加 label 欄位

        # get labels, features and drop unnecessary features
        labels = (df['label'] == 0).astype(int).values
        data = df[feature_names] # DataFrame
        features = feature_names.copy()

        # get categorical features and apply label encoding
        categorical_features = [f for f in features if data[f].dtype == 'O']
        category_map = {}
        for f in categorical_features:
            le = LabelEncoder()
            data_tmp = le.fit_transform(data[f].values)
            data[f] = data_tmp
            category_map[features.index(f)] = list(le.classes_)

        # only return data values
        data = data.values
        target_names = [0, 1]

        if return_X_y:
            return data, labels

        return Bunch(raw_data=raw_data, data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)


    def is_valid_path(self, seq):
        """
        判斷序列使否符合規則
        : param seq: List[str]，如 ['green', 'yellow']
        : output: boolean，True or False
        """
        valid = True # 預設為 True

        # 轉成 list of str（確保每個元素都是字串）
        if isinstance(seq, np.ndarray):
            seq = seq.tolist()
        seq = [str(s) for s in seq]  # 防止 float/int

        # 檢查規則
        if 'yellow' not in seq:  
            valid = False  # 必須有 yellow
        # if 'red' in simplified:
        #     valid = False  # 不能有 red

        # 檢查是否有 blue → yellow，且中間沒 green
        for i in range(1, len(seq)):
            if seq[i] == 'yellow':
                # 向前找最近的 blue
                for j in range(i - 1, -1, -1):
                    if seq[j] == 'green':
                        break  # 中間有 green，符合規則
                    if seq[j] == 'blue':
                        # 中間沒 green，違反規則
                        valid = False
                        break
            if not valid:
                break

        return int(valid)
    
robot_instance = RobotPredictor()