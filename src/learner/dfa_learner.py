"""
DFA (Deterministic Finite Automaton) Learner

This module contains:
1. DFA structure operations (from aalpy)
2. DFA manipulation functions (clone, merge, delete, etc.)
3. DFA learning algorithms
4. DFA visualization and export
"""
import gc
import itertools
import random
import re
import sys
import os
from typing import Tuple, Dict, List, Set, Any
from collections import defaultdict

from matplotlib import pyplot as plt
import numpy as np
from aalpy.automata.Dfa import Dfa, DfaState

from scar_rpni_size_capped_demo import learn_dfa_size_capped


# Add external_modules path for language module
_external_modules_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA'))
if _external_modules_path not in sys.path:
    sys.path.insert(0, _external_modules_path)
from language.explain import Language as ExplainLanguage

# 匯入優化模組
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
    from dfa_optimization import BatchPathChecker, MergeOptimizer, _cxp_cache, PerformanceMonitor
    OPTIMIZATION_AVAILABLE = True
    print("[INFO] DFA 優化模組已加載")
except ImportError as e:
    print(f"[WARNING] dfa_optimization module not found: {e}, running without optimizations")
    OPTIMIZATION_AVAILABLE = False
    BatchPathChecker = None
    MergeOptimizer = None
    _cxp_cache = None

from .base import BaseAutomataLearner


# ==============================================================
# DFA Operations (moved from dfa_operation.py)
# ==============================================================

def _alphabet_of(dfa) -> Set[str]:
    """Get the alphabet of the dfa"""
    if hasattr(dfa, "alphabet"):
        return set(dfa.alphabet)
    if hasattr(dfa, "get_input_alphabet"):
        try:
            return set(dfa.get_input_alphabet())
        except Exception:
            pass
    syms = set()
    for s in getattr(dfa, "states", []):
        syms.update(getattr(s, "transitions", {}).keys())
    return syms


def dfa_product(dfa1: Dfa, dfa2: Dfa, final_func) -> Dfa:
    """
    Generic DFA product construction (for intersection/union).
    final_func: function that takes (accept1, accept2) -> bool (accepting)
    """
    alphabet = list(_alphabet_of(dfa1) | _alphabet_of(dfa2))
    state_map: Dict[Tuple[str, str], DfaState] = {}
    queue: list = []

    def get_id(s1, s2):
        return f"({s1.state_id},{s2.state_id})"

    initial_tuple = (dfa1.initial_state, dfa2.initial_state)
    initial_id = get_id(*initial_tuple)
    initial_accept = final_func(initial_tuple[0].is_accepting, initial_tuple[1].is_accepting)
    initial = DfaState(initial_id, initial_accept)
    state_map[(initial_tuple[0], initial_tuple[1])] = initial
    queue.append(initial_tuple)

    while queue:
        curr1, curr2 = queue.pop(0)
        curr = state_map[(curr1, curr2)]
        for a in alphabet:
            if a not in curr1.transitions or a not in curr2.transitions:
                continue
            next1 = curr1.transitions[a]
            next2 = curr2.transitions[a]
            next_tuple = (next1, next2)
            if next_tuple not in state_map:
                accept = final_func(next1.is_accepting, next2.is_accepting)
                node = DfaState(get_id(next1, next2), accept)
                state_map[next_tuple] = node
                queue.append(next_tuple)
            curr.transitions[a] = state_map[next_tuple]
    all_states = list(state_map.values())
    return Dfa(initial, all_states)


def dfa_intersection(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    return dfa_product(dfa1, dfa2, lambda a1, a2: a1 and a2)


def dfa_union(dfa1: Dfa, dfa2: Dfa) -> Dfa:
    return dfa_product(dfa1, dfa2, lambda a1, a2: a1 or a2)


def get_base_dfa(learn_type, alphabet_map) -> Dfa:
    """Generate a DFA that sequentially moves from s0 to the end"""
    state_setup = {}
    steps = sorted(alphabet_map.keys())
    num_states = len(steps)

    for i, step in enumerate(steps):
        state_name = f's{i}'
        next_state = f's{i+1}'
        is_accepting = (i == num_states)
        transitions = {symbol: next_state for symbol in alphabet_map[step]}
        state_setup[state_name] = (is_accepting, transitions)

    final_state = f's{num_states}'
    state_setup[final_state] = (True, {})
    return Dfa.from_state_setup(state_setup)


def merge_linear_edges(dfa, learn_type=None):
    """
    Merge consecutive edges in the dfa.
    For TEXT type: if two consecutive edges are both wildcards (*), merge them.
    """
    def is_wildcard(sym):
        if sym == "*":
            return True
        if isinstance(sym, str) and re.match(r"p\d+_\(\*\)", sym):
            return True
        return False
    
    def get_position(sym):
        if not isinstance(sym, str):
            return None
        match = re.match(r"p(\d+)_", sym)
        return int(match.group(1)) if match else None
    
    states = set(dfa.states)
    changed = True
    while changed:
        changed = False
        for s in list(states):
            if s is dfa.initial_state or s.is_accepting:
                continue

            in_edges = []
            for p in states:
                for sym, q in p.transitions.items():
                    if q is s:
                        in_edges.append((p, sym))
                        if len(in_edges) > 1:
                            break
                if len(in_edges) > 1:
                    break

            out_edges = list(s.transitions.items())
            if len(in_edges) == 1 and len(out_edges) == 1:
                p, sym_in = in_edges[0] 
                sym_out, q = out_edges[0]

                if sym_in == sym_out:
                    p.transitions[sym_in] = q
                    states.remove(s)
                    dfa.states.remove(s)
                    changed = True
                    break
                
                if learn_type == "Text" and is_wildcard(sym_in) and is_wildcard(sym_out):
                    pos_in = get_position(sym_in)
                    pos_out = get_position(sym_out)
                    
                    if pos_in is not None and pos_out is not None and pos_out == pos_in + 1:
                        merged_label = f"p{pos_in}-{pos_out}_(*)"
                        del p.transitions[sym_in]
                        p.transitions[merged_label] = q
                        states.remove(s)
                        dfa.states.remove(s)
                        changed = True
                        break
                    
                    range_match = re.match(r"p(\d+)-(\d+)_\(\*\)", sym_in)
                    if range_match and pos_out is not None:
                        start_pos = int(range_match.group(1))
                        end_pos = int(range_match.group(2))
                        if pos_out == end_pos + 1:
                            merged_label = f"p{start_pos}-{pos_out}_(*)"
                            del p.transitions[sym_in]
                            p.transitions[merged_label] = q
                            states.remove(s)
                            dfa.states.remove(s)
                            changed = True
                            break
    return dfa


def merge_parallel_edges(dfa, learn_type):
    """Merge parallel edges in the dfa."""
    def clean_symbol(sym: str) -> str:
        if learn_type in ("Text", "Image"):
            match = re.search(r"\(([\d\.\-]+)\)", sym)
            if match:
                return match.group(1)
            return sym
        return sym

    def get_prefix(sym):
        if not isinstance(sym, str):
            return ""
        match = re.match(r"(p\d+)_\([\d\.\-\w]+\)", sym)
        return match.group(1) + "_" if match else ""

    def get_position(sym):
        if not isinstance(sym, str):
            return None
        match = re.match(r"p(\d+)_", sym)
        return match.group(1) if match else None

    alphabet = set(clean_symbol(sym) for sym in _alphabet_of(dfa))

    for s in dfa.states:
        target_map = {}
        new_transitions = {}
        prefix_map = {}
        position_map = {}

        for sym, t in s.transitions.items():
            clean_sym = clean_symbol(sym)
            new_transitions[clean_sym] = t
            if t != s:
                target_map.setdefault(t, set()).add(clean_sym)
                prefix = get_prefix(sym)
                if prefix:
                    prefix_map[t] = prefix
                
                if learn_type == "Text":
                    pos = get_position(sym)
                    if pos:
                        key = (t, pos)
                        position_map.setdefault(key, set()).add(sym)

        if learn_type == "Text":
            for (t, pos), symbols in position_map.items():
                if len(symbols) >= 2:
                    has_unk = any("UNK" in sym for sym in symbols)
                    has_word = any("UNK" not in sym for sym in symbols)
                    
                    if has_unk and has_word:
                        prefix = prefix_map.get(t, "")
                        wildcard = f"{prefix}(*)" if prefix else "*"
                        
                        for sym in symbols:
                            if sym in new_transitions:
                                del new_transitions[sym]
                        
                        new_transitions[wildcard] = t
            
            s.transitions = new_transitions
        else:
            for t, symbols in target_map.items():
                if symbols == alphabet:
                    s.transitions = new_transitions.copy()
                    for sym in list(symbols):
                        if sym in s.transitions and s.transitions[sym] is t:
                            del s.transitions[sym]

                    prefix = prefix_map.get(t, "")
                    if prefix:
                        s.transitions[f"{prefix}(*)"] = t
                    else:
                        s.transitions[f"*"] = t
    
    return dfa


def simplify_dfa(dfa, learn_type):
    """Simplify dfa by merging edges"""
    dfa = merge_parallel_edges(dfa, learn_type)
    dfa = merge_linear_edges(dfa, learn_type)
    return dfa


def scar_to_aalpy_dfa(scar_dfa) -> Dfa:
    """Convert scar_rpni_size_capped_demo.DFA to aalpy.automata.Dfa"""
    id2state = {}
    for q in sorted(scar_dfa.states, key=lambda x: str(x)):
        id2state[q] = DfaState(f"s{q}", q in scar_dfa.accepting)

    initial = id2state[scar_dfa.start]

    for (q, a), r in scar_dfa.delta.items():
        id2state[q].transitions[a] = id2state[r]

    return Dfa(initial, list(id2state.values()))


def dfa_intersection_any(d1, d2):
    """Intersection that handles both aalpy and scar DFAs"""
    if not isinstance(d1, Dfa):
        d1 = scar_to_aalpy_dfa(d1)
    if not isinstance(d2, Dfa):
        d2 = scar_to_aalpy_dfa(d2)
    return dfa_intersection(d1, d2)


def clone_dfa(dfa: Dfa) -> Dfa:
    """Deep clone a DFA"""
    old_to_new = {}

    for old_s in dfa.states:
        s_new = DfaState(old_s.state_id, is_accepting=old_s.is_accepting)
        s_new.prefix = []
        old_to_new[old_s] = s_new

    for old_s in dfa.states:
        for sym, tgt in old_s.transitions.items():
            if tgt not in old_to_new:
                ghost = DfaState(getattr(tgt, "state_id", str(tgt)), is_accepting=getattr(tgt, "is_accepting", False))
                ghost.prefix = []
                old_to_new[tgt] = ghost
            old_to_new[old_s].transitions[sym] = old_to_new[tgt]

    init_state = dfa.initial_state
    if isinstance(init_state, list):
        init_state = init_state[0]
    if init_state not in old_to_new:
        matched = next(
            (s for s in dfa.states if getattr(s, "state_id", None) == getattr(init_state, "state_id", None)),
            None
        )
        if matched:
            init_state = matched
        else:
            ghost = DfaState(getattr(init_state, "state_id", "q0"),
                            is_accepting=getattr(init_state, "is_accepting", False))
            old_to_new[init_state] = ghost

    return Dfa(states=list(old_to_new.values()), initial_state=old_to_new[init_state])


def remove_unreachable_states(dfa):
    """Remove unreachable states from DFA"""
    reachable = set()
    queue = [dfa.initial_state]
    while queue:
        s = queue.pop()
        if s not in reachable:
            reachable.add(s)
            for nxt in s.transitions.values():
                queue.append(nxt)
    dfa.states = [s for s in dfa.states if s in reachable]
    return dfa


def serialize_dfa(dfa) -> int:
    """Serialize DFA to hashable signature for deduplication"""
    items = []
    for s in sorted(dfa.states, key=lambda x: str(x.state_id)):
        trans = sorted([(sym, str(dst.state_id)) for sym, dst in s.transitions.items()])
        items.append((str(s.state_id), s.is_accepting, tuple(trans)))
    return hash(tuple(items))


def make_dfa_complete(dfa: Dfa, alphabet: list) -> Dfa:
    """Convert Partial DFA to Complete DFA by adding sink state"""
    sink_id = "sink_state"
    existing_ids = set(s.state_id for s in dfa.states)
    while sink_id in existing_ids:
        sink_id += "_"

    sink_state = DfaState(sink_id, is_accepting=False)
    
    for sym in alphabet:
        sink_state.transitions[sym] = sink_state

    added_sink = False
    for s in dfa.states:
        for sym in alphabet:
            if sym not in s.transitions:
                s.transitions[sym] = sink_state
                added_sink = True
    
    if added_sink:
        dfa.states.append(sink_state)
        
    return dfa


def trim_dfa(dfa):
    """Trim unreachable states from DFA"""
    alphabet = _alphabet_of(dfa)
    start = dfa.initial_state

    reachable = {start}
    queue = [start]

    while queue:
        cur = queue.pop(0)
        for a in alphabet:
            if a in cur.transitions:
                nxt = cur.transitions[a]
                if nxt not in reachable:
                    reachable.add(nxt)
                    queue.append(nxt)

    state_map = {}
    for old in reachable:
        new_state = DfaState(old.state_id)
        new_state.is_accepting = old.is_accepting
        new_state.transitions = {}
        state_map[old] = new_state
    
    for old, new in state_map.items():
        for a in alphabet:
            if a in old.transitions:
                nxt = old.transitions[a]
                if nxt in state_map:
                    new.transitions[a] = state_map[nxt]

    return Dfa(
        states=set(state_map.values()),
        initial_state=state_map[start],
    )

# ==============================================================
# DFA Visualization and Export
# ==============================================================
def dfa_to_mata(dfa, file_path):
    """Export DFA to Mata format for libMata"""
    def clean_state_name(name):
        name = str(name)
        name = ''.join(ch for ch in name if ch.isalnum())
        if not name.startswith("q"):
            name = "q" + name
        return name

    states = list(dfa.states)
    init_state = dfa.initial_state
    finals = [s for s in states if s.is_accepting]

    all_syms = sorted({sym for s in states for sym in s.transitions.keys()})
    symbol_map = {sym: i for i, sym in enumerate(all_syms)}
    state_map = {s.state_id: clean_state_name(s.state_id) for s in states}
    transition_map = {}

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("@NFA-explicit\n")
        f.write(f"%Initial {state_map[init_state.state_id]}\n")

        if finals:
            finals_str = " ".join(state_map[s.state_id] for s in finals)
            f.write(f"%Final {finals_str}\n")

        t_id = 0
        for s in states:
            src = state_map[s.state_id]
            for sym, tgt in s.transitions.items():
                tgt_name = state_map[tgt.state_id]
                sym_id = symbol_map[sym]
                f.write(f"{src} {sym_id} {tgt_name}\n")
                transition_map[t_id] = sym 
                t_id += 1
    return state_map, symbol_map, transition_map


def explain_axp_cxp(axps, cxps, symbol_map):
    """Print human-readable AXp and CXp explanations"""
    inv_map = {v: k for k, v in symbol_map.items()}
    for i, axp in enumerate(axps):
        print(f"AXp {i+1}: {[inv_map[x] for x in axp]}")
    for i, cxp in enumerate(cxps):
        print(f"Cxp {i+1}: {[inv_map[x] for x in cxp]}")


def get_test_word(mata_path, symbol_map):
    """Generate a test word from symbol map"""
    grouped = defaultdict(list)
    path = []

    for k, v in symbol_map.items():
        match = re.match(r"(p\d+)_", k)
        if not match:
            continue
        pos = match.group(1)
        grouped[pos].append((k, v))

    for pos in sorted(grouped.keys(), key=lambda x: int(x[1:])): 
        symbols = grouped[pos]
        for k, v in symbols:
            if "(1.0)" in k or "*" in k:
                path.append((pos, v))
                break 

    path_ids = [v for _, v in path]
    # print("Path:", path_ids)
    return path_ids


# ==============================================================
# DFA Learner Class
# ==============================================================
class DFASampler:
    """
    Eeuse the function of TabularSampler，only change the perturbation function to DFA perturbation method
    """

    def __init__(self, predictor, base_sampler: None, alphabet: List = [], seed: int = None, edit_distance: int = 1):
        self.predictor = predictor
        self.tab_sampler = base_sampler
        self.alphabet = alphabet
        self.edit_distance = edit_distance

        self.instance_label = None
        self.instance = None
        self.n_covered_ex = 10
        self._built = True
        
        if seed is not None:
            random.seed(seed)
        else:
            self.seed = seed

    def set_instance_label(self, X):
        self.tab_sampler.set_instance_label(X)
        self.instance = X

    def set_n_covered(self, n):
        self.n_covered_ex = n
        self.tab_sampler.set_n_covered(n)

    def build_lookups(self, data=None):
        """AnchorTabular 預期 sampler 有 build_lookups"""
        self._built = True
        return ({}, {}, {})

    def compare_labels(self, samples: np.ndarray):
        return self.tab_sampler.compare_labels(samples)
    
    def perturbation(self, num_samples: int):
        symbols = self.alphabet if self.alphabet else list(set(self.instance))
        edit_distance = self.edit_distance
        max_trials = num_samples * 10
        
        local_paths = []
        
        # sampling
        local_trials = 0 # avoid infinite loop
        while len(local_paths) < int(num_samples) and local_trials < max_trials:
            local_trials += 1
            if local_trials >= max_trials:
                break

            new_instance = list(self.instance)
            op = random.choice(["replace", "insert", "delete"])
            max_edit = min(edit_distance, len(new_instance))
            edit_dist = random.randint(0, max_edit) if max_edit > 0 else 0

            if op == "replace":
                if len(new_instance) > 0:
                    replace_indices = random.sample(range(len(new_instance)), edit_dist)
                    for idx in replace_indices:
                        new_instance[idx] = random.choice([s for s in symbols if not np.array_equal(s, new_instance[idx])])

            elif op == "insert":
                for _ in range(edit_dist):
                    insert_idx = random.randint(0, len(new_instance))
                    new_instance.insert(insert_idx, random.choice(symbols))

            elif op == "delete":
                if len(new_instance) > 0:
                    delete_indices = sorted(random.sample(range(len(new_instance)), min(edit_dist, len(new_instance))), reverse=True)
                    for idx in delete_indices:
                        del new_instance[idx]

            hashable_instance = tuple(
                tuple(x) if isinstance(x, np.ndarray) else x
                for x in new_instance
            )
            local_paths.append(hashable_instance)

        d_samples = local_paths
        return local_paths, d_samples

    def __call__(self, num_samples, compute_labels=True):
        raw_data, d_raw_data = self.perturbation(num_samples)

        if compute_labels:
            labels = self.compare_labels(raw_data)

            # variable-length
            if getattr(self.tab_sampler, "d_train_data", np.array([])).dtype == object:
                covered_true = [seq for seq, lab in zip(raw_data, labels) if lab == self.instance_label][:self.n_covered_ex]
                covered_false = [seq for seq, lab in zip(raw_data, labels) if lab != self.instance_label][:self.n_covered_ex]
                covered_true = np.array(covered_true, dtype=object)
                covered_false = np.array(covered_false, dtype=object)
            else:
                covered_true = raw_data[labels][:self.n_covered_ex]
                covered_false = raw_data[np.logical_not(labels)][:self.n_covered_ex]

            return [raw_data, labels.astype(int)]
        else:
            return [d_raw_data]
        
class DFALearner(BaseAutomataLearner):
    """DFA (Deterministic Finite Automaton) Learner"""
    
    def __init__(self):
        super().__init__()
    
    # ========== Implement Abstract Methods ==========
    def get_sampler(self):
        from learner.dfa_learner import DFASampler
        return DFASampler

    def check_path_accepted(self, dfa, path) -> bool:
        """Check if path is accepted by DFA"""
        dfa = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        dfa.reset_to_initial()
        for symbol in path:
            try:
                dfa.step(symbol)
            except KeyError:
                return False
        return dfa.current_state.is_accepting
    
    def check_path_exist(self, dfa, path) -> bool:
        """Check if path exists in DFA"""
        dfa = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        dfa.current_state = dfa.initial_state
        for symbol in path:
            try:
                dfa.step(symbol)
            except KeyError:
                return False
        return True
    
    def get_accept_paths(self, dfa, max_depth=50) -> List[List[Any]]:
        """Find all accepting paths of the DFA using DFS"""
        dfa_obj = dfa[0] if isinstance(dfa, (list, tuple)) else dfa
        paths = set() 

        def dfs(state, prefix, visited):
            if len(prefix) > max_depth:
                return
            if getattr(state, "is_accepting", False):
                paths.add(tuple(prefix))
            for sym, next_state in state.transitions.items():
                if (id(next_state), sym) not in visited:
                    dfs(next_state, prefix + [sym], visited | {(id(next_state), sym)})

        dfs(dfa_obj.initial_state, [], set())
        return [list(p) for p in sorted(paths, key=len)]
    
    def clone_automaton(self, dfa):
        """Deep clone a DFA"""
        old_to_new = {}

        for old_s in dfa.states:
            s_new = DfaState(old_s.state_id, is_accepting=old_s.is_accepting)
            s_new.prefix = []
            old_to_new[old_s] = s_new

        for old_s in dfa.states:
            for sym, tgt in old_s.transitions.items():
                if tgt not in old_to_new:
                    ghost = DfaState(getattr(tgt, "state_id", str(tgt)), is_accepting=getattr(tgt, "is_accepting", False))
                    ghost.prefix = []
                    old_to_new[tgt] = ghost
                old_to_new[old_s].transitions[sym] = old_to_new[tgt]

        init_state = dfa.initial_state
        if isinstance(init_state, list):
            init_state = init_state[0]
        if init_state not in old_to_new:
            matched = next(
                (s for s in dfa.states if getattr(s, "state_id", None) == getattr(init_state, "state_id", None)),
                None
            )
            if matched:
                init_state = matched
            else:
                ghost = DfaState(getattr(init_state, "state_id", "q0"),
                                is_accepting=getattr(init_state, "is_accepting", False))
                old_to_new[init_state] = ghost

        return Dfa(states=list(old_to_new.values()), initial_state=old_to_new[init_state])
    
    def serialize_automaton(self, dfa) -> int:
        """Serialize DFA to hashable signature for deduplication"""
        items = []
        for s in sorted(dfa.states, key=lambda x: str(x.state_id)):
            trans = sorted([(str(sym), str(dst.state_id)) for sym, dst in s.transitions.items()])
            items.append((str(s.state_id), s.is_accepting, tuple(trans)))
        return hash(tuple(items))
    
    def automaton_to_graphviz(self, dfa, filename=None, show_sink=False, instance=None, output_dir="output") -> str:
        """
        Convert DFA to Graphviz DOT string and save visualization.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to visualize
        filename : str
            Output filename
        show_sink : bool
            Whether to show sink states
        instance : list or tuple, optional
            Original instance to highlight its path in the DFA with color
        output_dir : str
            Output directory
            
        Returns
        -------
        str
            DOT content string
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        def clean_state_name(name):
            name = str(name)
            return ''.join(ch if ch.isalnum() or ch == '_' else '_' for ch in name)

        def clean_label(label):
            return str(label).replace('"', "'")

        # draw path edges if instance is provided
        path_edges = set()
        if instance is not None:
            current = dfa.initial_state
            for i, symbol in enumerate(instance):
                if symbol in current.transitions:
                    next_state = current.transitions[symbol]
                    path_edges.add((clean_state_name(current.state_id), clean_state_name(next_state.state_id), clean_label(symbol)))
                    current = next_state
                else:
                    break

        lines = ["digraph DFA {", "  rankdir=LR;", '  node [shape=circle];']
        start_name = clean_state_name(dfa.initial_state.state_id)
        lines.append('  __start__ [shape=point];')
        lines.append(f'  __start__ -> "{start_name}";')

        for state in dfa.states:
            if not show_sink and hasattr(dfa, "sink") and state == dfa.sink:
                continue
            shape = "doublecircle" if state.is_accepting else "circle"
            lines.append(f'  "{clean_state_name(state.state_id)}" [shape={shape}];')

        for state in dfa.states:
            for symbol, next_state in state.transitions.items():
                if not show_sink and hasattr(dfa, "sink") and (
                    state == dfa.sink or next_state == dfa.sink
                ):
                    continue
                src_name = clean_state_name(state.state_id)
                dst_name = clean_state_name(next_state.state_id)
                label_name = clean_label(symbol)
                
                if (src_name, dst_name, label_name) in path_edges:
                    lines.append(f'  "{src_name}" -> "{dst_name}" [label="{label_name}", color=red, penwidth=2.5, fontcolor=red];')
                else:
                    lines.append(f'  "{src_name}" -> "{dst_name}" [label="{label_name}"];')

        lines.append("}")
        
        # Save to file
        dot_content = "\n".join(lines)
        if filename:
            dot_path = os.path.join(output_dir, filename)
            with open(dot_path, "w") as f:
                f.write(dot_content)

        return dot_content


    def create_automata_sized(self, positive_samples, negative_samples, alphabet):
        """Create DFA with size-capped learning"""
        print(f'\nPassive learning sample count: {len(positive_samples + negative_samples)}\n')

        results = []
        Ms = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        for M in Ms:
            out = learn_dfa_size_capped(
                positive_samples,
                negative_samples,
                alphabet,
                M,
                include_sink_in_count=False,
                verbose=True,
                beam_enabled=True
            )
            states = out["total_states_including_sink"] if False else out["non_sink_states"]
            results.append((M, states, out["train_accuracy"]))
            print(f"M={M:>3} | states={states:>3} | train_acc={out['train_accuracy']:.3f}")
            
        Ms_list = [m for m, _, _ in results]
        accs = [acc for _, _, acc in results]
        plt.figure()
        plt.plot(Ms_list, accs, marker='o')
        plt.xlabel("Size cap M (non-sink states)" if not False else "Size cap M (including sink)")
        plt.ylabel("Training accuracy")
        plt.title("SCAR-RPNI: accuracy vs. size cap")
        plt.grid(True)
        plt.show()
        return out['dfa']
    
    def create_init_automata(self, data_type, positive_samples, negative_samples):
        """Create initial DFA using RPNI algorithm"""
        from aalpy.learning_algs import run_RPNI
        init_passive_data = []
        
        if data_type != 'Tabular':
            sample_source = positive_samples if positive_samples else negative_samples
            text_length = len(sample_source[0]) if sample_source else 0
            alphabet_map = {}
            
            all_samples = positive_samples + negative_samples
            for pos in range(text_length):
                symbols_at_pos = set()
                for sample in all_samples:
                    if pos < len(sample):
                        symbols_at_pos.add(sample[pos])
                alphabet_map[pos] = list(symbols_at_pos)
        
        for i in positive_samples:
            init_passive_data.append([tuple(i), True])
        for i in negative_samples:
            init_passive_data.append([tuple(i), False])
        dfa = run_RPNI(init_passive_data, automaton_type='dfa', print_info=False)
        # print("dfa", dfa)
        return dfa
    
    def perturbation(self, num_samples, max_trials=500):
        """
        Randomly perturb sequences to generate initial passive samples:
        - replace: replace n symbols
        - insert: insert n new symbols
        - delete: delete n symbols
        """
        perturbed = set()
        trials = 0

        while len(perturbed) < num_samples and trials < max_trials:
            new_instance = self.instance.copy()
            op = random.choice(["replace", "append", "delete"])
            edit_dist = min(self.edit_distance, len(new_instance))  # 確保不超過序列長度

            if op == "replace":
                replace_indices = random.sample(range(len(new_instance)), edit_dist)
                for idx in replace_indices:
                    new_instance[idx] = random.choice([s for s in self.symbols[0] if s != new_instance[idx]])

            elif op == "append":
                replace_indices = random.sample(range(len(new_instance)), edit_dist)
                for idx in replace_indices:
                    insert_idx = random.randint(0, len(new_instance))
                    new_instance.insert(insert_idx, random.choice(self.symbols[0]))

            elif op == "delete":
                replace_indices = random.sample(range(len(new_instance)), edit_dist)
                delete_indices = sorted(random.sample(range(len(new_instance)), edit_dist), reverse=True)
                for idx in delete_indices:
                    del new_instance[idx]
            perturbed.add(tuple(new_instance))
            trials += 1

        perturbed = list(perturbed)
        if len(perturbed) < num_samples:
            extra = random.choices(perturbed, k=num_samples - len(perturbed))
            perturbed.extend(extra)

        return list(perturbed)
    
    def score_fn(self, dfa, state):
        """
        Score function for evaluating DFA quality
        """
        dfa_id = id(dfa)
        acc = state['t_positives'].get(dfa_id, 0) + state['t_negatives'].get(dfa_id, 0)
        n = state['t_nsamples'].get(dfa_id, 1)
        acc_rate = acc / n if n > 0 else 0
        num_states = len(getattr(dfa, 'states', []))
        return (acc_rate, -num_states)

    def _propose_delete(self, dfa, state, data, labels, seen_signatures, beam_size):
        """
        Propose new DFAs by deleting states from the given DFA.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new DFAs created by deletion
        """
        import heapq, gc
        heap = []
        new_dfas = []
        for s in list(dfa.states):
            # Skip initial state and the last accepting state
            if s == dfa.initial_state or (s.is_accepting and sum(x.is_accepting for x in dfa.states) <= 1):
                continue
            
            # new_dfa = self.clone_automaton(dfa)
            new_dfa = dfa.copy()
            target_state = next(x for x in new_dfa.states if x.state_id == s.state_id)
            
            # Inline delete logic
            print(f"Deleting state {target_state.state_id}")
            outgoing = dict(target_state.transitions)
            for st in list(new_dfa.states):
                for sym, next_s in list(st.transitions.items()):
                    if next_s == target_state:
                        if sym in outgoing:
                            st.transitions[sym] = outgoing[sym]
                        else:
                            st.transitions[sym] = st
            
            if target_state in new_dfa.states:
                new_dfa.states.remove(target_state)

            for st in new_dfa.states:
                for sym, nxt in list(st.transitions.items()):
                    if nxt not in new_dfa.states:
                        st.transitions[sym] = st

            # Remove unreachable states before checking for accepting states
            remove_unreachable_states(new_dfa)
            
            # check if there are still accepting states after deletion
            if not any(st.is_accepting for st in new_dfa.states):
                print(f"  [SKIP] Deleting {target_state.state_id} leaves no accepting state, skipping.")
                del new_dfa, target_state, outgoing
                gc.collect()
                continue

            sig = self.serialize_automaton(new_dfa)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                self.update_state_metrics(state, dfa, new_dfa, data, labels, "DELETE")
                new_dfas.append(new_dfa)
            else:
                del new_dfa, target_state, outgoing, st
                gc.collect()

        del s
        gc.collect()
        return new_dfas

    # def get_merge_candidates(self, dfa, data, labels, similarity_threshold=0.7):
    #     """
    #     Return pairs of states with similar future behavior (forward signature similarity >= threshold).
    #     """
    #     import itertools

    #     # Precompute forward signatures
    #     sigs = {
    #         s: self.compute_forward_signature(dfa, s)
    #         for s in dfa.states
    #     }

    #     candidates = []
    #     for s1, s2 in itertools.combinations(dfa.states, 2):
    #         if s1 == dfa.initial_state or s2 == dfa.initial_state:
    #             continue

    #         sig1, sig2 = sigs[s1], sigs[s2]
    #         if not sig1 or not sig2:
    #             continue

    #         sim = len(sig1 & sig2) / len(sig1 | sig2)
    #         if sim >= similarity_threshold:
    #             candidates.append((s1, s2))

    #     return candidates

    # def compute_forward_signature(self, dfa, state, max_depth=3, max_samples=30):
    #     """
    #     Approximate future behavior of a state by probing short suffixes.
    #     Returns a frozenset of (word, accept/reject).
    #     """
    #     import random
    #     alphabet = list(_alphabet_of(dfa))
    #     signature = set()

    #     for _ in range(max_samples):
    #         length = random.randint(1, max_depth)
    #         word = tuple(random.choice(alphabet) for _ in range(length))

    #         cur = state
    #         for sym in word:
    #             if sym not in cur.transitions:
    #                 break
    #             cur = cur.transitions[sym]
    #         else:
    #             signature.add((word, cur.is_accepting))

    #     return frozenset(signature)

    def check_merge_feasible_rule(self, s1, s2, state_label_dist, critical_states):
        # 1. DFA semantics
        if s1.is_accepting != s2.is_accepting:
            return False

        dist1 = state_label_dist[s1.state_id]
        dist2 = state_label_dist[s2.state_id]
        main_label1 = max(dist1, key=dist1.get, default=None)
        main_label2 = max(dist2, key=dist2.get, default=None)

        # 2. Only protect critical states
        if s1 in critical_states and s2 in critical_states:
            if main_label1 != main_label2:
                return False

        return True


    # def is_boundary_state(self, state_id, state_label_dist, purity_threshold=0.9):
    #     dist = state_label_dist[state_id]
    #     total = sum(dist.values())
    #     if total == 0:
    #         return False
    #     return max(dist.values()) / total < purity_threshold
    
    # def collect_feasible_merge_pairs(self, dfa, state_label_dist, critical_states): 
    #     """ 預先收集所有 merge feasible 的 state pair 回傳 [(s1, s2), ...] """ 
    #     pairs = [] 
    #     states = list(dfa.states) 
    #     for s1, s2 in itertools.combinations(states, 2): 
    #         if s1 == s2 or s1 == dfa.initial_state or s2 == dfa.initial_state: 
    #             continue 
    #         if self.check_merge_feasible_rule(s1, s2, state_label_dist, critical_states): 
    #             pairs.append((s1, s2)) 
    #     return pairs
    
    # def is_rare_but_pure_state(self, state_id, state_support, state_label_dist, rarity_threshold=0.05, purity_threshold=0.8): 
    #     """ 判斷該 state 是否為「極少數但純」的 minority state。
    #     - support 低於 rarity_threshold 
    #     - purity 高於 purity_threshold """ 
    #     total_samples = sum(state_support.values()) 
    #     support = state_support[state_id] 
    #     dist = state_label_dist[state_id] 
    #     if total_samples == 0 or not dist: 
    #         return False 
    #     if support / total_samples > rarity_threshold:
    #         return False 
    #     if max(dist.values()) / sum(dist.values()) < purity_threshold: 
    #         return False 
    #     return True
    
    def collect_merge_pairs_simple(self, dfa, data, labels, max_pairs=20):
        """
        優先合併 label 分布一致的狀態
        """
        from collections import defaultdict
        
        #  label 分布
        state_label_dist = defaultdict(lambda: defaultdict(int))
        for seq, y in zip(data, labels):
            cur = dfa.initial_state
            for sym in seq:
                if sym not in cur.transitions:
                    break
                cur = cur.transitions[sym]
            state_label_dist[cur.state_id][y] += 1
        
        # 計算每個狀態的主要 label
        def main_label(state_id):
            dist = state_label_dist[state_id]
            return max(dist, key=dist.get) if dist else None
        
        # 選擇 label 相同的狀態對（非常快）
        pair_scores = []
        for s1, s2 in itertools.combinations(dfa.states, 2):
            if s1 == dfa.initial_state or s2 == dfa.initial_state:
                continue
            if s1.is_accepting != s2.is_accepting:
                continue
            
            # 同類 label 優先
            if main_label(s1.state_id) == main_label(s2.state_id):
                pair_scores.append((1.0, s1, s2))
            else:
                dist1 = state_label_dist[s1.state_id]
                dist2 = state_label_dist[s2.state_id]
                # Jaccard 相似度
                all_labels = set(dist1.keys()) | set(dist2.keys())
                inter = sum(min(dist1.get(y,0), dist2.get(y,0)) for y in all_labels)
                union = sum(max(dist1.get(y,0), dist2.get(y,0)) for y in all_labels)
                sim = inter / union if union > 0 else 0
                if sim > 0:
                    pair_scores.append((sim, s1, s2))
        
        pair_scores.sort(reverse=True, key=lambda x: x[0])
        
        return [(s1, s2) for _, s1, s2 in pair_scores[:max_pairs]]
    
    def collect_merge_pairs_all(self, dfa, max_pairs=None):
        """
        嘗試所有狀態對，讓 accuracy filter 決定
        """
        pairs = []
        for s1, s2 in itertools.combinations(dfa.states, 2):
            if s1 == dfa.initial_state or s2 == dfa.initial_state:
                continue
            if s1.is_accepting == s2.is_accepting:
                pairs.append((s1, s2))
        
        return pairs[:max_pairs] if max_pairs else pairs

    def _propose_merge(self, dfa, state, data, labels, seen_signatures, beam_size):
        """
        Propose new DFAs by merging pairs of states in the given DFA.
        Uses intelligent scoring to prioritize high-impact merges.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
            
        Returns
        -------
        list
            List of new DFAs created by merging states
        """
        import heapq, gc
        # heap = []
        new_dfas = []

        # 使用高效的多維度評分來選擇最佳合併候選對
        feasible_pairs = self.collect_merge_pairs_simple(dfa, data, labels, max_pairs=20)
        # feasible_pairs = [(s1, s2) for s1, s2 in itertools.combinations(list(dfa.states) , 2)]

        if not feasible_pairs:
            print("  [MERGE] 沒有找到可行的合併狀態對")
            return new_dfas

        for s1, s2 in feasible_pairs:
            # do merge
            new_dfa = dfa.copy()
            s1_new = next(x for x in new_dfa.states if x.state_id == s1.state_id)
            s2_new = next(x for x in new_dfa.states if x.state_id == s2.state_id)

            print(f"Merging state {s2_new.state_id} into {s1_new.state_id}")
            for st in list(new_dfa.states):
                for sym, nxt in list(st.transitions.items()):
                    if nxt == s2_new:
                        st.transitions[sym] = s1_new

            for sym, nxt in s2_new.transitions.items():
                s1_new.transitions[sym] = nxt
            
            s1_new.is_accepting = s1_new.is_accepting or s2_new.is_accepting
            if isinstance(new_dfa.states, set):
                new_dfa.states.discard(s2_new)
            else:
                try:
                    new_dfa.states.remove(s2_new)
                except ValueError:
                    pass

            # Remove unreachable states before checking for accepting states
            remove_unreachable_states(new_dfa)

            # check if there are still reachable accepting states after merge
            if not any(st.is_accepting for st in new_dfa.states):
                print(f"  [SKIP] Merging {s2.state_id} into {s1.state_id} leaves no accepting state, skipping.")
                del new_dfa, s1_new, s2_new
                gc.collect()
                continue

            sig = self.serialize_automaton(new_dfa)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                self.update_state_metrics(state, dfa, new_dfa, data, labels, "MERGE")
                new_dfas.append(new_dfa)
            else:
                del new_dfa, s1_new, s2_new
                gc.collect()

        gc.collect()
        return new_dfas

    def _aggregate_cxp_analysis(self, dfa, data, labels, misclassified_paths, alphabet_map, dfa_sig, top_k=10, max_samples=50):
        """
        Aggregate CXP analysis over multiple misclassified paths to compute blame scores.
        
        This method:
        1. Collects paths through DFA for each misclassified sample
        2. Computes CXP (Contrastive eXplanation Path) for each sample
        3. Counts edge occurrences and CXP hits for each edge
        4. Computes blame score for each edge
        5. Returns top-K blamed edges sorted by blame score
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to analyze
        data : list
            Full dataset (used for checking paths)
        labels : np.ndarray
            Labels for dataset
        misclassified_paths : list
            List of misclassified sequences (error samples)
        alphabet_map : dict
            Mapping from symbols to alphabet indices for CXP computation
        dfa_sig : int
            Serialized signature of DFA for caching CXP results
        top_k : int
            Number of blamed edges to return (default: 10)
        max_samples : int
            Maximum number of samples to process for efficiency (default: 50)
            
        Returns
        -------
        dict
            Contains:
            - 'blamed_edges': list of (src_state_id, symbol, blame_score) tuples, sorted descending
            - 'edge_stats': dict of {(src_state_id, symbol): {'occ': count, 'cxp': count}}
            - 'total_cxp_computed': number of CXP computations performed
            
        Notes
        -----
        - Blame score formula: blame(edge) = cxp_hits(edge) / (occ(edge) + 1)
        - The +1 in denominator prevents division by zero
        - CXP results are filtered to keep only shortest explanations (most reliable)
        - Path indices are mapped directly to CXP indices for hit counting
        """
        from dfa_optimization import _cxp_cache
        
        edge_occ = defaultdict(int)  # (src_state_id, symbol) -> count of paths using this edge
        edge_cxp = defaultdict(int)  # (src_state_id, symbol) -> count of CXP hits on this edge
        cxp_computed = 0

        # Limit samples for efficiency
        if max_samples and len(misclassified_paths) > max_samples:
            misclassified_paths = random.sample(misclassified_paths, max_samples)
        
        print("Computing CXP...")
        # For selected misclassified path, compute path and CXP
        for idx, test_word in enumerate(misclassified_paths):
            # Trace path through DFA
            current = dfa.initial_state
            path = []  # List of (state_id, symbol, next_state_id)
            for sym in test_word:
                next_state = current.transitions.get(sym)
                if next_state is None:
                    break
                # Store state_id instead of state objects
                path.append((current.state_id, sym, next_state.state_id))
                edge_occ[(current.state_id, sym)] += 1
                current = next_state
            
            if not path:
                continue
            
            # Compute CXP for this word
            try:
                # Encode word for this iteration
                encoded_word = [alphabet_map[sym] for sym in test_word if sym in alphabet_map]
                if not encoded_word:
                    continue
                
                # Define compute function that captures encoded_word for this iteration
                def compute_cxp_for_word():
                    """Compute CXP for a single word."""
                    explanation_engine = ExplainLanguage()
                    return explanation_engine.explain_word(
                        "dfa_explicit.mata",
                        from_mata=True,
                        word=encoded_word,
                        ascii=encoded_word,
                        target_axp=False,
                        bootstrap_cxp_size_1=False,
                        print_exp=False,
                    )
                
                # Use cache if available
                if OPTIMIZATION_AVAILABLE and _cxp_cache is not None:
                    result = _cxp_cache.get_cxp(dfa_sig, test_word, compute_cxp_for_word)
                else:
                    result = compute_cxp_for_word()
                cxp_computed += 1
                
                cxp_raw = result.get("cxps", [])
                if not cxp_raw:
                    continue
                
                # for each misclassified sample, select shortest explanations
                if cxp_raw:
                    min_len = min(len(seq) for seq in cxp_raw)
                    cxp_raw = [seq for seq in cxp_raw if len(seq) == min_len]
                
                # Aggregate CXP hits
                # CXP returns indices of positions in the original word
                for seq in cxp_raw:
                    # seq is a list of indices into test_word, need to map to path
                    for pos_in_word in seq:
                        if 0 <= pos_in_word < len(path):
                            src_state_id, sym, _ = path[pos_in_word]
                            edge_cxp[(src_state_id, sym)] += 1
            
            except Exception as e:
                print(f"[Warning] CXP computation failed for sample {idx}: {e}")
                continue
            
        # Compute blame score: blame(e) = cxp[e] / (occ[e] + 1)
        blame_scores = []
        for edge, occ_count in edge_occ.items():
            cxp_count = edge_cxp[edge]
            blame_score = cxp_count / (occ_count + 1.0)  # +1 to avoid division by zero
            blame_scores.append((edge[0], edge[1], blame_score, occ_count, cxp_count))
        
        # Sort by blame score descending
        blame_scores.sort(key=lambda x: x[2], reverse=True)
        print(f"Computed CXP for {cxp_computed} samples, found {len(blame_scores)} blamed edges")

        # Return top-K
        result = {
            'blamed_edges': [(src, sym, score) for src, sym, score, _, _ in blame_scores[:top_k]],
            'edge_stats': {(src, sym): {'occ': occ, 'cxp': cxp} 
                          for src, sym, _, occ, cxp in blame_scores},
            'total_cxp_computed': cxp_computed
        }
        return result

    def _propose_delta(self, dfa, state, data, labels, seen_signatures, top_k_blamed_edges=20, max_misclassified_samples=50):
        """
        Propose new DFAs by modifying transitions based on aggregated CXP analysis.
        
        Uses CXP aggregation over multiple misclassified paths to identify the most
        problematic transitions (edges with high blame score), then proposes fixes.
        
        Parameters
        ----------
        dfa : Dfa
            The DFA to modify
        state : dict
            State dictionary for tracking metrics
        data : list
            Training data
        labels : np.ndarray
            Labels for training data
        seen_signatures : set
            Set of already seen DFA signatures to avoid duplicates
        top_k_blamed_edges : int
            Number of top blamed edges to try modifying (default: 10)
        max_misclassified_samples : int
            Maximum number of misclassified samples to process (default: 50)
            
        Returns
        -------
        list
            List of new DFAs created by transition modification
        """
        new_dfas = [dfa]  # Always include the original DFA
        
        # Identify misclassified samples
        if OPTIMIZATION_AVAILABLE and BatchPathChecker is not None:
            accepts = BatchPathChecker.check_paths_batch(dfa, data)
        else:
            accepts = np.array([self.check_path_accepted(dfa, p) for p in data])
        
        false_accept_indices = np.where((labels == 0) & (accepts == True))[0]
        true_reject_indices = np.where((labels == 1) & (accepts == False))[0]
        
        misclassified_indices = np.concatenate([false_accept_indices, true_reject_indices])
        if len(misclassified_indices) == 0:
            print("No misclassified samples found, skipping.")
            return new_dfas
        
        misclassified_paths = [data[i] for i in misclassified_indices]
        print(f"Found {len(misclassified_paths)} misclassified samples (false_accept={len(false_accept_indices)}, true_reject={len(true_reject_indices)})")
        
        # Export DFA to mata format for CXP computation
        try:
            _, alphabet_map, _ = dfa_to_mata(dfa, "dfa_explicit.mata")
            dfa_sig = self.serialize_automaton(dfa)
        except Exception as e:
            print(f"Failed to export DFA to mata: {e}")
            return new_dfas
        
        # Aggregate CXP analysis
        try:
            cxp_result = self._aggregate_cxp_analysis(
                dfa, data, labels, misclassified_paths, alphabet_map, dfa_sig,
                top_k=top_k_blamed_edges,
                max_samples=max_misclassified_samples
            )
            blamed_edges = cxp_result['blamed_edges']
        except Exception as e:
            print(f"CXP aggregation failed: {e}")
            return new_dfas
        
        if not blamed_edges:
            print("No blamed edges found after CXP aggregation.")
            return new_dfas
        
        # Build reachability information
        reverse_adj = defaultdict(set)
        for st in dfa.states:
            for _, nxt in st.transitions.items():
                reverse_adj[nxt].add(st)
        
        can_reach_accepting = set()
        bfs_q = [st for st in dfa.states if st.is_accepting]
        while bfs_q:
            st = bfs_q.pop()
            if st not in can_reach_accepting:
                can_reach_accepting.add(st)
                for prev in reverse_adj[st]:
                    bfs_q.append(prev)
        
        orig_state_by_id = {st.state_id: st for st in dfa.states}
        
        # Only process the highest blamed edge
        if not blamed_edges:
            print("No blamed edges to process")
            return new_dfas
        
        src_state_id, symbol, blame_score = blamed_edges[0]  # Only take the top blamed edge
        src_state = orig_state_by_id.get(src_state_id)
        if src_state is None or symbol not in src_state.transitions:
            print(f"Cannot process top blamed edge: {src_state_id} --{symbol}-->")
            return new_dfas
        
        old_target = src_state.transitions[symbol]
        print(f"Processing highest blamed edge: {src_state_id} --{symbol}--> {old_target.state_id} (blame={blame_score:.4f})")
        
        # Try redirecting this edge to different target states
        for target_state in dfa.states:
            if target_state.state_id == old_target.state_id:
                continue  # Skip if no change
            
            new_dfa = dfa.copy()
            src_new = next(x for x in new_dfa.states if x.state_id == src_state_id)
            target_new = next(x for x in new_dfa.states if x.state_id == target_state.state_id)
            
            old_target_new = src_new.transitions[symbol]
            src_new.transitions[symbol] = target_new
            
            # Check if accepting states are still reachable
            if old_target_new in can_reach_accepting:
                reachable = set()
                queue = [new_dfa.initial_state]
                while queue:
                    st = queue.pop()
                    if st not in reachable:
                        reachable.add(st)
                        for nxt in st.transitions.values():
                            queue.append(nxt)
                
                if not any(st.is_accepting for st in reachable):
                    # Revert if no accepting states reachable
                    del new_dfa
                    gc.collect()
                    continue
            
            remove_unreachable_states(new_dfa)
            
            # Check for accepting states
            if not any(st.is_accepting for st in new_dfa.states):
                del new_dfa
                gc.collect()
                continue
            
            sig = self.serialize_automaton(new_dfa)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                self.update_state_metrics(
                    state, dfa, new_dfa, data, labels, 
                    f"DELTA-AGG(blame={blame_score:.2f})"
                )
                new_dfas.append(new_dfa)
                print(f"→ Generated new candidate: {src_state_id} --{symbol}--> {target_state.state_id}")
            else:
                del new_dfa
                gc.collect()
        
        print(f"Generated {len(new_dfas) - 1} new candidate DFAs")
        return new_dfas

    def propose_automata(self, dfas, state, iteration, previous_best: list, output_dir: str, beam_size: int = 10):
        """
        Propose new DFA candidates by expanding existing DFAs.
        
        This method coordinates three strategies:
        - DELETE: Remove states from DFA (even iterations)
        - MERGE: Combine pairs of states (even iterations)
        - DELTA: Modify transitions based on CXP analysis (odd iterations)
        
        Parameters
        ----------
        dfas : list
            List of current DFAs
        state : dict
            State dictionary for tracking metrics
        sample_fcn : object
            Sampling function with feature values
        iteration : int
            Current iteration number
        previous_best : list
            List of best DFAs from previous iteration
        data_type : str
            Type of data ('Tabular', 'Text', etc.)
            
        Returns
        -------
        list
            List of proposed new DFAs
        """
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]

        # Initialize metrics for first iteration
        if iteration == 0:
            for dfa in dfas:
                t_idx = set(i for i, p in enumerate(data) if self.check_path_exist(dfa, p))
                accepts = np.array([self.check_path_accepted(dfa, p) for p in data])
                true_accept = np.sum((labels == 1) & (accepts == True))
                false_reject = np.sum((labels == 0) & (accepts == False))
                
                dfa_id = id(dfa)
                state['t_idx'][dfa_id] = t_idx
                state['t_nsamples'][dfa_id] = float(len(data))
                state['t_accepted'][dfa_id] = float(np.sum(accepts))
                state['t_positives'][dfa_id] = float(true_accept)
                state['t_negatives'][dfa_id] = float(false_reject)
                state['t_order'][dfa_id].append(dfa_id)

                print("--------------------------------------------")
                print(f"Proposed DFA ID: {dfa_id}")
                # print(self.automaton_to_graphviz(dfa, filename="initial_dfa", output_dir=output_dir))
            
            return dfas

        seen_signatures = set()
        new_dfas = []

        for dfa in previous_best:
            if iteration % 2 == 0:
                # Even iterations: DELETE and MERGE
                new_dfas.extend(self._propose_delete(dfa, state, data, labels, seen_signatures, beam_size))
                new_dfas.extend(self._propose_merge(dfa, state, data, labels, seen_signatures, beam_size))
            else:
                # Odd iterations: DELTA
                new_dfas.extend(self._propose_delta(dfa, state, data, labels, seen_signatures))
        
        unique_dfas = []
        seen = set()
        for dfa in new_dfas:
            sig = self.serialize_automaton(dfa)
            if sig not in seen:
                seen.add(sig)
                unique_dfas.append(dfa)
            else:
                del dfa
                gc.collect()
        return unique_dfas
        # return new_dfas


# ==============================================================
# Module-level exports for backward compatibility
# ==============================================================

__all__ = [
    # Learner class
    'DFALearner',
    # DFA operations
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
    # 'recompute_coverage',
    'delete_state',
    'merge_states',
    'serialize_dfa',
    'make_dfa_complete',
    'trim_dfa',
    # Path checking
    'check_path_exist',
    'check_path_accepted',
    'get_accept_paths',
    # Visualization
    'dfa_to_graphviz',
    'dfa_to_mata',
    'explain_axp_cxp',
    'get_test_word',
    # Helper
    '_alphabet_of',
]
