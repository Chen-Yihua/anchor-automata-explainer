from collections import defaultdict
from automaton.utils import add_position_to_sample

def check_conflict_sample(pos_sample, neg_sample):
    """
    Check if there are samples that are both positive and negative in the passive samples
    : param pos_sample: List[List[str]], positive samples with label 1
    : param neg_sample: List[List[str]], negative samples with label 0
    """
    pos_lookup = defaultdict(list)
    neg_lookup = defaultdict(list)
    for p in pos_sample:
        pos_lookup[tuple(p)].append(p)
    for n in neg_sample:
        neg_lookup[tuple(n)].append(n)
    positive_set_check = set(pos_lookup.keys())
    negative_set_check = set(neg_lookup.keys())

    # find intersection
    conflict_samples = set(positive_set_check & negative_set_check)

    # print conflicting samples
    if conflict_samples:
        for item in conflict_samples:
            print(f"\nConflicting sample (sorted): {item}")
        raise ValueError(
            f"Conflicting labeled examples detected: {list(conflict_samples)!r}. "
            "Please change the DFA or fix the labels.")    

def get_testing_samples(learn_type, state):
    """
    Get all coverage samples (the initial batch of samples)
    : param learn_type: str, learning type ('Image', 'Text', 'Tabular')
    : param state: dict, learning record
    : return: List, coverage samples
    """
    if learn_type == 'Image' or learn_type == 'Text':
        testing_samples = list(state['coverage_data'])
        testing_samples = add_position_to_sample(testing_samples)
    if learn_type == 'Tabular':
        testing_samples = list(state['coverage_raw'])
    return testing_samples

def get_positive_test_samples(learn_type, coverage_data, coverage_label, coverage_raw=None):
    """
    Return samples with coverage_label==1 based on learn_type:
      - learn_type == 'Tabular' → return rows in coverage_raw labeled as 1
      - others → return rows in coverage_data labeled as 1
    """
    import numpy as np
    y = np.asarray(coverage_label).ravel()
    if learn_type == 'Tabular':
        X = np.asarray(coverage_raw)
    else:
        X = np.asarray(coverage_data)

    mask = (y == 1)
    selected = X[mask]

    if learn_type != 'Tabular':
        samples = [list(row) if isinstance(row, (list, tuple, np.ndarray)) else [row] for row in selected.tolist()]
        return add_position_to_sample(samples)
    
    return selected