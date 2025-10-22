# automaton/learner.py
from .data import check_conflict_sample, rebalance_samples, get_testing_samples, get_positive_test_samples
from .utils import add_position_to_sample, get_alphabet_from_samples, append_eos
from .utils import add_position_to_sample, check_path_exist, check_path_accepted, dfa_to_graphviz

class AutomatonLearner:
    def __init__(self):
        self.model = None
        self.training_data = [] # RPNI 格式的訓練資料
        self.testing_data = [] # 測試資料
        self.pos_samples = [] # 正例樣本
        self.neg_samples = [] # 負例樣本

    def log_samples(self, learn_type, raw_data, labels, binary_data):
        """
        紀錄訓練過程中的抽樣樣本
        """
        for i in range(len(raw_data)):
            if learn_type == 'Image' or learn_type == 'Text':
                if labels[i] == 1.0:
                    self.pos_samples.append(binary_data[0][i].tolist())
                else:
                    self.neg_samples.append(binary_data[0][i].tolist())
            if learn_type == 'Tabular':
                if labels[i] == 1.0:
                    self.pos_samples.append(raw_data[i].tolist())
                else:
                    self.neg_samples.append(raw_data[i].tolist())
        return
    
    def convert_to_rpni_format(self, pos_samples=None, neg_samples=None):
        """
        轉換資料格式為 RPNI learning 可用的樣本格式，並加入 training_data
        : param pos_samples: anchor 學習過程中的正例
        : param neg_samples: anchor 學習過程中的負例
        : output: input sequences and corresponding label，如 [[(i1,i2,i3, ...), label], ...]
        """
        for i in pos_samples:
            self.training_data.append([tuple(i), True])
        if neg_samples == None: 
            return self.training_data
        for i in neg_samples:
            self.training_data.append([tuple(i), False])
        return self.training_data
    
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

    def collect_samples(self):
        """
        取得目前紀錄的正例、負例樣本
        : return: pos_samples, neg_samples
        """
        return self.pos_samples, self.neg_samples
    
    def create_automata(self, learn_type, anchor, state):
        """
        取得自動機
        : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
        : param state: 學習紀錄
        : return: dfa
        """
        positive_samples, negative_samples = self.pos_samples, self.neg_samples
        if learn_type != 'Tabular':
            positive_samples = add_position_to_sample(positive_samples)
            negative_samples = add_position_to_sample(negative_samples)
        check_conflict_sample(positive_samples, negative_samples) # 檢查衝突樣本
        self.convert_to_rpni_format(positive_samples, negative_samples) # 轉換樣本格式並存入 training_data
        print(f'\n被動學習樣本數量: {len(positive_samples + negative_samples)}\n')

        alphabet = get_alphabet_from_samples(positive_samples , negative_samples)
        # results = []
        # Ms = [22, 23, 24, 25, 26]
        M = 6
        # for M in Ms:
        dfa = self.learn_dfa(positive_samples, negative_samples, alphabet, M)
        self.get_evaluation(learn_type, anchor, state, dfa, positive_samples, negative_samples) # 計算指標
            # _, _, _, dfa_test_precision = self.get_evaluation(learn_type, anchor, state, dfa) # 計算指標
            # results.append((M, len(dfa.states), dfa_test_precision))
        # plot_precision_vs_dfa_size(results) # 繪製圖表
        return dfa, self.testing_data
           
    def learn_dfa(self, positive_data, negative_data, alphabet, M):
        """
        被動學習 DFA
        """
        from scar_rpni_size_capped_demo import learn_dfa_size_capped
        from aalpy.learning_algs import run_RPNI, run_Alergia
        # self.model = run_Alergia(passive_data, automaton_type='mc', eps=0.05, print_info=True)
        self.model = run_RPNI(self.training_data, automaton_type='dfa', print_info=False)
        # print("DFA:", self.model)
        
        # out = learn_dfa_size_capped(
        #     positive_data,
        #     negative_data,
        #     alphabet,
        #     M,
        #     include_sink_in_count=False,
        #     verbose=True,
        # )
        # self.model = out["dfa"]
        # dot = dfa_to_graphviz(self.model, show_sink=True)
        # print("DFA:", dot)
        return self.model
    
    def get_evaluation(self, learn_type, anchor, state, dfa, positive_samples, negative_samples):
        """
        計算 anchor 和 automata 的指標
        : param learn_type: str，學習類型 ('Image', 'Text', 'Tabular')
        : param state: 學習紀錄
        """
        import numpy as np
        self.testing_data = get_testing_samples(learn_type, state) 
        training_samples = positive_samples + negative_samples
        # exist_train_samples = [sample for sample in training_samples if check_path_exist(dfa, sample)== True]
        # exist_test_samples = [sample for sample in testing_samples if check_path_exist(dfa, sample)== True]
        accepted_train_samples = [sample for sample in training_samples if check_path_accepted(dfa, sample)== True]
        accepted_test_samples = [sample for sample in self.testing_data if check_path_accepted(dfa, sample)== True]
        labeled_one_train_samples = [sample for sample in training_samples if (sample in positive_samples)]
        labeled_one_test_samples = get_positive_test_samples(learn_type, state['coverage_data'], state['coverage_label'], state['coverage_raw'])
        accepted_train_one_samples = [sample for sample in labeled_one_train_samples if check_path_accepted(dfa, sample)== True]
        accepted_test_one_samples = [sample for sample in labeled_one_test_samples if check_path_accepted(dfa, sample)== True]
        coverage_label = state['coverage_label']
        t_coverage_idx = state['t_coverage_idx'][anchor]
        covered_labels = [coverage_label[i] for i in t_coverage_idx]
        num_positive = sum(1 for label in covered_labels if label == 1)

        print(f"Anchor 訓練集 Coverage: {state['t_nsamples'][anchor] / len(self.testing_data) if len(self.testing_data) > 0 else 0.0:.4f}")
        print(f"Anchor 測試集 Coverage: {(len(state['t_coverage_idx'][anchor]) / len(state['coverage_data'])) if len(state['coverage_data']) > 0 else 0.0 :.4f}")
        print(f"Anchor 訓練集 Precision: {(state['t_positives'][anchor] / state['t_nsamples'][anchor]):.4f}")
        # print(f"Anchor 測試集 Precision: {(len(state['t_covered_true'][anchor]) / (len(state['t_covered_true'][anchor]) + (len(state['t_covered_false'][anchor])))) :.4f}")
        print(f"Anchor 測試集 Precision: {((num_positive) / ((len(state['t_coverage_idx'][anchor])))) if len(state['t_coverage_idx'][anchor]) > 0 else 0.0 :.4f}")

        # 計算 DFA 訓練集 Coverage
        dfa_train_coverage = len(accepted_train_samples) / len(training_samples) if len(training_samples) > 0 else 0
        print(f"Automaton 訓練集 Coverage: {dfa_train_coverage:.4f}")

        # 計算 DFA 測試集 Coverage
        dfa_test_coverage = len(accepted_test_samples) / len(self.testing_data) if len(self.testing_data) > 0 else 0
        print(f"Automaton 測試集 Coverage: {dfa_test_coverage:.4f}")

        # 計算 DFA 訓練集 Precision
        dfa_train_precision = len(accepted_train_one_samples) / len(accepted_train_samples) if len(accepted_train_samples) > 0 else 0
        print(f"Automaton 訓練集 Precision: {dfa_train_precision:.4f}")

        # 計算 DFA 測試集 Precision
        dfa_test_precision = len(accepted_test_one_samples) / len(accepted_test_samples) if len(accepted_test_samples) > 0 else 0
        print(f"Automaton 測試集 Precision: {dfa_test_precision:.4f}\n")    

        return dfa_train_coverage, dfa_test_coverage, dfa_train_precision, dfa_test_precision
    
AUTO_INSTANCE = AutomatonLearner()