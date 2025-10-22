from collections import defaultdict
from .utils import add_position_to_sample, append_eos

def check_conflict_sample(pos_sample, neg_sample):
    """
    查看被動樣本中是否有同時為正例與反例的樣本
    : param pos_sample: List[List[str]]，label 為 1 的正例樣本
    : param neg_sample: List[List[str]]，label 為 0 的反例樣本
    """
    pos_lookup = defaultdict(list)
    neg_lookup = defaultdict(list)

    # 填入樣本
    for p in pos_sample:
        pos_lookup[tuple(p)].append(p)
    for n in neg_sample:
        neg_lookup[tuple(n)].append(n)

    # 找重複（用排序後樣本作為 key）
    positive_set_check = set(pos_lookup.keys())
    negative_set_check = set(neg_lookup.keys())

    # 找交集
    conflict_samples = set(positive_set_check & negative_set_check)

    # 印出衝突樣本
    if conflict_samples:
        for item in conflict_samples:
            print(f"\n衝突樣本（sorted）: {item}")
        raise ValueError(
            f"Conflicting labeled examples detected: {list(conflict_samples)!r}. "
            "Please change the DFA or fix the labels.")    

def rebalance_samples(S_pos, S_neg, strategy="oversample_minor", seed=0):
    import random
    rnd = random.Random(seed)
    n_pos, n_neg = len(S_pos), len(S_neg)
    if n_pos == 0 or n_neg == 0:
        return S_pos, S_neg
    
    if strategy == "undersample_major":
        if n_neg > n_pos:
            S_neg = rnd.sample(S_neg, n_pos)
        else:
            S_pos = rnd.sample(S_pos, n_neg)

    elif strategy == "oversample_minor":
        if n_pos < n_neg:
            S_pos = S_pos + rnd.choices(S_pos, k=n_neg - n_pos)
        else:
            S_neg = S_neg + rnd.choices(S_neg, k=n_pos - n_neg)

    return S_pos, S_neg

def get_testing_samples(learn_type, state):
    """
    取出所有覆蓋樣本 (初始抽樣的那批樣本)
    : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
    : param state: dict，學習紀錄
    : return: List，覆蓋樣本
    """
    if learn_type == 'Image' or learn_type == 'Text':
        testing_samples = list(state['coverage_data'])
        testing_samples = add_position_to_sample(testing_samples)
    # testing_samples = append_eos(testing_samples)  
    if learn_type == 'Tabular':
        testing_samples = list(state['coverage_raw'])

    return testing_samples

def get_positive_test_samples(learn_type, coverage_data, coverage_label, coverage_raw=None):
    """
    依 learn_type 回傳 coverage_label==1 的樣本：
      - learn_type == 'Tabular' → 回傳 coverage_raw 中被標為 1 的列
      - 其他 → 回傳 coverage_data 中被標為 1 的列
    """
    import numpy as np
    y = np.asarray(coverage_label).ravel()
    if learn_type == 'Tabular':
        X = np.asarray(coverage_raw)
    else:
        X = np.asarray(coverage_data)

    mask = (y == 1)
    selected = X[mask]

    if learn_type != 'Tabular': # 轉成 list[list] 再做位置編碼
        samples = [list(row) if isinstance(row, (list, tuple, np.ndarray)) else [row] for row in selected.tolist()]
        return add_position_to_sample(samples)
    
    return selected

# def get_alergia_samples(learn_type, anchor_idx, mab, sampler=None):
    #     """
    #     取出適用 alergia 的被動學習樣本
    #     """
    #     precision_samples_idx = []
    #     if learn_type == 'Image' or learn_type == 'Text':
    #         precision_samples_idx = mab.state['t_idx'][tuple(anchor_idx)] 
    #     if learn_type == 'Tabular':
    #         binary_anchor_raw = [] # 取得 anchor 的二元編碼
    #         for bin_idx, raw_idx in sampler.enc2feat_idx.items():
    #             if raw_idx in anchor_idx:
    #                 binary_anchor_raw.append(bin_idx)
    #         precision_samples_idx = mab.state['t_idx'][tuple(binary_anchor_raw)] 
    #     precision_samples = [mab.state['data'][i] for i in precision_samples_idx] # 根據樣本索引取出 binary sample
    #     return precision_samples