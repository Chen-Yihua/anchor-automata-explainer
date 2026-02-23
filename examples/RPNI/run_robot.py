
import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
EXTERNAL_MODULES = os.path.join(PROJECT_ROOT, 'external_modules')
MODIFIED_MODULES = os.path.join(PROJECT_ROOT, 'modified_modules')
EXPLAINING_FA = os.path.join(EXTERNAL_MODULES, 'Explaining-FA')
INTERPRETERA_SRC = os.path.join(EXTERNAL_MODULES, 'interpretera', 'src')

# Put modified_modules at the front so 'alibi' resolves to local version
for path in [MODIFIED_MODULES, SRC_PATH, EXTERNAL_MODULES, EXPLAINING_FA, INTERPRETERA_SRC, PROJECT_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)


from tee import Tee
import numpy as np
import pickle
from modified_modules.alibi.explainers import AnchorTabular
from data.dataset_loader import fetch_custom_dataset
from robot_operation import robot_instance


# 載入資料
b = fetch_custom_dataset( # robot_randomlen
    source="datasets/robot_randomlen.csv",
    mode="tabular",
    target_col="label",
    return_X_y=False
)
data = b.data
raw_data = b.raw_data
target = b.target
feature_names = b.feature_names # ['s0','s1',...]
category_map = b.category_map # {col_idx: [class_names...]}
alphabet = ['blue', 'yellow', 'green']
automaton_type = 'DFA'  # 'DFA' or 'RA'

# predict function
def robot_predict_fn(X):
    """
    Predict function for the robot dataset
    : param X: List[List], input samples
    : return: np.array, predictions (0/1)
    """
    X = np.asarray(X, dtype=object)
    preds = []
    for row in X:
        seq = []
        for j, v in enumerate(row):
            try:
                v_int = int(v)
                if j in category_map and 0 <= v_int < len(category_map[j]):
                    seq.append(category_map[j][v_int])
                else:
                    seq.append(str(v_int))
            except (ValueError, TypeError):
                seq.append(str(v))
        preds.append(int(robot_instance.is_valid_path(seq)))
    return np.array(preds, dtype=int)

# fit anchor explainer
explainer = AnchorTabular(
    predictor=robot_predict_fn,
    feature_names=feature_names,
    categorical_names=dict(category_map),
    seed=1
)
explainer.fit(
    automaton_type=automaton_type, 
    train_data=data,
    alphabet=alphabet
)
explainer.samplers[0].d_train_data = data

# explain sentence
test_instance = ['blue', 'blue', 'blue', 'yellow']
# explainer parameters
learn_type = 'Tabular'
coverage_samples = 5000
batch_size = 1000
n_covered_ex = 1000
min_samples_start = 1000
accuracy_threshold = 0.95
state_threshold = 5
tau = 0.01
delta = 0.01
# epsilon_stop = 0.01
edit_distance = 3
beam_size = 1

with open("TestRobot1.txt", "w", encoding="utf-8") as log_file:
    sys.stdout = Tee(sys.stdout, log_file) 
    print("\nInstance: %s" % test_instance)
    print("Prediction: %s" % robot_predict_fn([test_instance]))
    print("\n============Training step============")

    explanation = explainer.explain(
        learn_type,
        automaton_type=automaton_type,
        alphabet=alphabet,
        X=test_instance,
        edit_distance=edit_distance,
        coverage_samples=coverage_samples,
        batch_size=batch_size,
        n_covered_ex=n_covered_ex,
        min_samples_start=min_samples_start,
        accuracy_threshold=accuracy_threshold,
        state_threshold=state_threshold,
        tau=tau,
        delta=delta,
        beam_size=beam_size,
        verbose=True,
    )

    # Anchor result
    print('\n==============Result==============')
    print('Automata:', explanation.automata)
    print('Training Accuracy:', explanation.training_accuracy)
    print('Testing Accuracy:', explanation.testing_accuracy)
    print('State:', explanation.state)
    
    mab = explainer.mab # learning record

    # generate automaton
    # origin_dfa, testing_data = AUTO_INSTANCE.create_automata_sized(mab.type, alphabet=alphabet)
    # print(f"Origin DFA:{origin_dfa}")

    # find AXp, CXp
    # test_word = [int(v) for v in data[1]]
    # print("test_word:", test_word)
    # state_map, alphabet_map = dfa_to_mata(mab.dfa, "dfa_explicit.mata")
    # test_word = get_test_word("dfa_explicit.mata", alphabet_map) # get test path
    # explanation_engine = Language()
    # result = explanation_engine.explain_word(
    #     "dfa_explicit.mata",
    #     from_mata=True,
    #     word=[1, 1, 2, 2, 1],
    #     ascii=False,
    #     target_axp=True,
    #     bootstrap_cxp_size_1=False,
    #     print_exp=True
    # )
    # explain_axp_cxp(result["axps"], result["cxps"], alphabet_map) # convert axps, cxps to text
    
    # calculate DFA Intersection
    # features = explanation.raw['feature'] # extract anchor values
    # alphabet_map = {} # build alphabet mapping for dfa
    # for i in mab.sample_fcn.feature_values:
    #     if i not in alphabet_map:
    #         alphabet_map[i] = []
    #     for j in range(len(mab.sample_fcn.feature_values[i])):
    #         alphabet_map[i].append(j)
    # sub_dfa = get_base_dfa(learn_type, alphabet_map, features, test_instance)
    # inter_dfa = dfa_intersection_any(origin_dfa, sub_dfa) # intersection

    # pickle save dfa, testing data (without position, robot)
    # result = []
    # for idx in range(len(mab.testing_data)):
    #     data = list(mab.testing_data[idx])
    #     result.append((data, mab.state['coverage_label'][idx]))

    # with open("robot_dfa.pkl", "wb") as f:
    #     pickle.dump(inter_dfa, f)
    # with open("robot_testing_data.pkl", "wb") as f:
    #     pickle.dump(result, f)

    # simplify DFA
    # inter_dfa.make_input_complete()
    # inter_dfa.minimize()
    # # print("intersection dfa:", inter_dfa)
    # final_dfa = simplify_dfa(inter_dfa, learn_type) # merge edges
    # print(f"Final DFA:{final_dfa}\n")

    sys.stdout = sys.__stdout__