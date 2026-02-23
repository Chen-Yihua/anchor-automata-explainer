"""
Automaton module - Automaton structures and operations

NOTE: Most functionality has been moved to the learner module.
This file re-exports for backward compatibility.
"""

from aalpy.automata.Dfa import Dfa as DFA, DfaState as DFAState

# Import from learner module (where the code now lives)
from learner.dfa_learner import (
    # DFA operations
    clone_dfa,
    # delete_state,
    # merge_states,
    serialize_dfa,
    dfa_intersection,
    dfa_union,
    dfa_intersection_any,
    get_base_dfa,
    simplify_dfa,
    merge_linear_edges,
    merge_parallel_edges,
    scar_to_aalpy_dfa,
    remove_unreachable_states,
    # recompute_coverage,
    make_dfa_complete,
    trim_dfa,
    _alphabet_of,    
    # DFA visualization
    dfa_to_mata,
    explain_axp_cxp,
    get_test_word,
)

# from learner.ra_learner import (
#     # RA operations
#     clone_ra,
#     serialize_ra,
#     ra_to_graphviz,
#     check_ra_path_accepted,
#     check_ra_path_exist,
#     # merge_ra_states,  # inlined into _propose_merge
#     # delete_ra_state,  # inlined into _propose_delete
# )

from .utils import (
    # General utilities (not DFA-specific)
    add_position_to_sample,
    tokenize_sentence,
    plot_beam_stats,
    plot_dfa_beam_stats,  # backward compatibility
)

__all__ = [
    # DFA
    'DFA',
    'DFAState',
    # Register Automaton
    'RegisterAutomaton',
    'RAState',
    # DFA Operations
    'clone_dfa',
    # 'delete_state',  # inlined into _propose_delete
    # 'merge_states',  # inlined into _propose_merge
    'serialize_dfa',
    'dfa_intersection',
    'dfa_union',
    'dfa_intersection_any',
    'get_base_dfa',
    'simplify_dfa',
    'merge_linear_edges',
    'merge_parallel_edges',
    'scar_to_aalpy_dfa',
    'remove_unreachable_states',
    # 'recompute_coverage',
    'make_dfa_complete',
    'trim_dfa',
    '_alphabet_of',
    # DFA Path Checking
    'check_path_accepted',
    'check_path_exist',
    'get_accept_paths',
    # DFA Visualization
    'dfa_to_graphviz',
    'dfa_to_mata',
    'explain_axp_cxp',
    'get_test_word',
    # RA Operations
    'clone_ra',
    'serialize_ra',
    'ra_to_graphviz',
    'check_ra_path_accepted',
    'check_ra_path_exist',
    # 'merge_ra_states',  # inlined into _propose_merge
    # 'delete_ra_state',  # inlined into _propose_delete
    # General Utils
    'add_position_to_sample',
    'tokenize_sentence',
    'plot_beam_stats',
    'plot_dfa_beam_stats',
]
