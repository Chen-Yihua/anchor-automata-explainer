# Learner module - Learning algorithms for automata
from .base import BaseAutomataLearner
from .dfa_learner import (
    DFALearner,
    # DFA operations (for backward compatibility)
    dfa_product,
    dfa_intersection,
    dfa_union,
    dfa_intersection_any,
    get_base_dfa,
    merge_linear_edges,
    merge_parallel_edges,
    simplify_dfa,
    scar_to_aalpy_dfa,
    clone_dfa,
    remove_unreachable_states,
    serialize_dfa,
    make_dfa_complete,
    trim_dfa,
    dfa_to_mata,
    explain_axp_cxp,
    get_test_word,
    _alphabet_of,
)
from .ra_learner import (
    RegisterAutomataLearner,    
)
from .factory import LearnerFactory, get_learner, AUTO_INSTANCE

__all__ = [
    # Base
    'BaseAutomataLearner',
    # DFA Learner
    'DFALearner', 
    # RA Learner
    'RegisterAutomataLearner',
    # Factory
    'LearnerFactory',
    'get_learner',
    'AUTO_INSTANCE',
    # DFA Operations
    'dfa_product',
    'dfa_intersection',
    'dfa_union',
    'dfa_intersection_any',
    'get_base_dfa',
    'merge_linear_edges',
    'merge_parallel_edges',
    'simplify_dfa',
    'scar_to_aalpy_dfa',
    'clone_dfa',
    'remove_unreachable_states',
    'serialize_dfa',
    'make_dfa_complete',
    'trim_dfa',
    'check_path_exist',
    'check_path_accepted',
    'get_accept_paths',
    'dfa_to_graphviz',
    'dfa_to_mata',
    'explain_axp_cxp',
    'get_test_word',
    '_alphabet_of',
]
