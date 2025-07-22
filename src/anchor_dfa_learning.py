from aalpy.learning_algs import run_RPNI, run_Alergia
from collections import defaultdict

class AutomatonLearner:
    def __init__(self):
        self.model = None
        self.data = []

    def convert_to_rpni_format(self, pos_samples=None, neg_samples=None):
        """
        轉換資料格式為 RPNI learning 可用的樣本格式
        : param positive_sample: List[List[str]]，anchor 學習過程中的正例
        : param negative_sample: List[List[str]]，anchor 學習過程中的負例
        : output: input sequences and corresponding label，如 [[(i1,i2,i3, ...), label], ...]
        """      
        for i in pos_samples:
            self.data.append([tuple(i), True])
        if neg_samples == None: 
            return self.data
        for i in neg_samples:
            self.data.append([tuple(i), False])
        return self.data
    
    # def convert_to_alergia_format(self, pos_samples=None, neg_samples=None):
    #     """
    #     轉換資料格式為 Alergia learning 可用的樣本格式
    #     : param positive_sample: List[List[str]]，anchor 學習過程中的正例
    #     : param negative_sample: List[List[str]]，anchor 學習過程中的負例
    #     : output: List[List[str]]
    #     """
    #     self.data = []
    #     for samples in pos_samples:
    #         samples = list(samples)    
    #         samples.insert(0,"start") # 加入開始符號
    #         self.data.append(samples)
    #     if neg_samples == None: 
    #         return self.data
    #     for samples in neg_samples:
    #         samples = list(samples)  
    #         samples.insert(0,"start")
    #         self.data.append(samples)        
    #     return self.data
    
    def check_conflict_sample(self, pos_sample, neg_sample):
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
        conflict_samples = positive_set_check & negative_set_check

        # 印出衝突樣本
        if conflict_samples:
            for item in conflict_samples:
                print(f"\n衝突樣本（sorted）: {item}")
                print(f" 正例版本: {pos_lookup[item]}")
                print(f" 反例版本: {neg_lookup[item]}")
        else:
            print("沒有重複樣本")
    
    def learn_dfa(self, passive_data):
        """
        被動學習 DFA
        """
        # self.model = run_Alergia(passive_data, automaton_type='mc', eps=0.05, print_info=True)
        self.model = run_RPNI(passive_data, automaton_type='dfa', print_info=False)
        return self.model
    
# def calculate_path_probability(self, sample):
#     """
#     計算路徑的機率
#     : param sample: List[str]，某條路徑
#     : output: float，該條路徑的機率
#     """
#     prob = 1.0
#     current_state = 'start'

#     for symbol in sample[1:]:
#         is_exist = False
#         for state in self.model.states:
#             if state.output == current_state: # 找出對應 output 的 state
#                 for (next_state, p) in state.transitions: 
#                     if next_state.output == symbol: # 找出對應 output 的 state
#                         prob *= p
#                         current_state = next_state.output # 更新 current_state
#                         is_exist = True
#                         break
#                 break
#             else:
#                 is_exist = False
#         if not is_exist: 
#             return 0.0 # 沒有該路徑
#     return prob     
     
def log_samples(raw_data, labels, binary_data):
    """
    紀錄訓練過程中的抽樣樣本 (包含二元值、原始值與其對應 label)
    """
    samples_set = []
    for i in range(len(raw_data)):
        samples_set.append({
            'binary_sample': binary_data[0][i].tolist(),
            'label': labels[i],
            'original_sample': raw_data[i].tolist(),
        })
    return samples_set

def check_path_exist(dfa, path):
    """
    看 path 是否存在 DFA
    """
    dfa.current_state = dfa.initial_state
    for symbol in path:
        try:
            dfa.step(symbol)
        except KeyError:
            return False
    return True

def check_path_accepted(dfa, path):
    """
    檢查 path 是否被 DFA 接受
    """
    dfa.reset_to_initial()
    for symbol in path:
        try:
            dfa.step(symbol)
        except KeyError:
            return False
    return dfa.current_state.is_accepting

def get_rpni_samples(learn_type, mab):
    """
    取出適用 rpni 的被動學習樣本 (當前所有抽樣樣本)
    """
    positive_samples = []
    negative_samples = []
    if learn_type == 'Image' or learn_type == 'Text':
        for _, samples in mab.anchor_analysis_log.items():
            for sample in samples:
                if sample['label'] == 1.0:
                    positive_samples.append(sample['binary_sample'])
                else:
                    negative_samples.append(sample['binary_sample'])
    if learn_type == 'Tabular':
        for _, samples in mab.anchor_analysis_log.items():
            for sample in samples:
                if sample['label'] == 1.0:
                    positive_samples.append(sample['original_sample'])
                else:
                    negative_samples.append(sample['original_sample'])
    return positive_samples, negative_samples

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

def add_position_to_sample(samples):
    """
    將每個樣本特徵加上位置資訊
    """
    return [[f'p_({i},{v})' for i, v in enumerate(sample)] for sample in samples]

def get_precision_samples(learn_type, anchor_idx, mab):
    """
    取得訓練過程中符合某 anchor 條件的抽樣樣本
    : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
    : param anchor_idx: List[int]，當前 anchor 的索引
    : param mab: AutomatonLearner 實例，包含學習紀錄 (state)
    : return: List，符合 anchor 條件的抽樣樣本
    """
    precision_samples_idx = []
    precision_samples = []
    if learn_type == 'Image' or learn_type == 'Text':
        precision_samples_idx = mab.state['t_idx'][tuple(anchor_idx)] 
        precision_samples = [mab.state['data'][i] for i in precision_samples_idx] # 根據樣本索引取出 binary sample
    if learn_type == 'Tabular':
        samples_list = mab.anchor_analysis_log[tuple(anchor_idx)]
        for sample in samples_list:
            precision_samples.append(sample['original_sample'])

    return precision_samples

def get_covered_samples(learn_type, state):
    """
    取出所有覆蓋樣本 (初始抽樣的那批樣本)
    : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
    : param state: dict，學習紀錄
    : return: List，覆蓋樣本
    """
    if learn_type == 'Image' or learn_type == 'Text':
        covered_samples = state['coverage_data']
    if learn_type == 'Tabular':
        covered_samples = state['coverage_raw']

    return covered_samples

def get_covered_anchor_samples(learn_type, anchor_idx, mab):
    """
    取出符合 anchor 的覆蓋樣本 (初始抽樣的那批樣本)
    : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
    : param state: dict，學習紀錄
    : return: List，覆蓋樣本
    """
    covered_samples = []
    for idx in mab.state['t_coverage_idx'][tuple(anchor_idx)]:
        if learn_type == 'Image' or learn_type == 'Text':
            covered_samples.append(mab.state['coverage_data'][idx])
        if learn_type == 'Tabular':
            covered_samples.append(mab.state['coverage_raw'][idx])

    return covered_samples

def get_accuracy_samples(learn_type, mab):
    """
    取得所有覆蓋樣本中的正反例
    : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
    : param mab: AutomatonLearner 實例，包含學習紀錄 (state)
    : return: List，覆蓋樣本
    """
    if learn_type == 'Image' or learn_type == 'Text':
        coverage_data = mab.state['coverage_data']
    else:
        coverage_data = mab.state['coverage_raw']

    coverage_labels = mab.state['coverage_label']
    mask_true = (coverage_labels == 1)
    covered_true = coverage_data[mask_true]
    covered_true = add_position_to_sample(covered_true)
    mask_false = (coverage_labels == 0)
    covered_false = coverage_data[mask_false]
    covered_false = add_position_to_sample(covered_false)        
    return covered_true, covered_false

# 建立全域單例實例
AUTO_INSTANCE = AutomatonLearner()

