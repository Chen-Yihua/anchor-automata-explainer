from typing import Tuple, Dict
from aalpy.automata.Dfa import Dfa, DfaState

def dfa_product(dfa1: Dfa, dfa2: Dfa, final_func) -> Dfa:
    """
    Generic DFA product construction (for intersection/union).
    final_func: function that takes (accept1, accept2) -> bool (accepting)
    """
    alphabet = list(set(dfa1.get_input_alphabet()) | set(dfa2.get_input_alphabet()))

    # States are pairs of (dfa1_state, dfa2_state)
    state_map: Dict[Tuple[str, str], DfaState] = {}
    queue: list = []

    def get_id(s1, s2):
        return f"({s1.state_id},{s2.state_id})"

    # Initial state
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
            # follow transitions, if undefined, skip
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


def get_base_dfa(alphabet_map={}):
    """
    生成一個從 s0 依序走到尾的自動機
    """
    state_setup = {}

    steps = sorted(alphabet_map.keys())  # 確保順序正確
    num_states = len(steps)

    for i, step in enumerate(steps):
        state_name = f's{i}'
        next_state = f's{i+1}' if i < num_states - 1 else f's{i+1}'
        is_accepting = (i == num_states)  # 最後一個是接受狀態

        transitions = {f'p_({i},{symbol})': next_state for symbol in alphabet_map[step]}
        state_setup[state_name] = (is_accepting, transitions)

    # 最後一個狀態（sink）為接受狀態，無 transition
    final_state = f's{num_states}'
    state_setup[final_state] = (True, {})

    dfa = Dfa.from_state_setup(state_setup)
    return dfa


def merge_linear_edges(dfa):
    """
    合併 dfa 中連續的 edge
    """
    # 取出所有 state
    states = set(dfa.states)
    changed = True

    while changed:
        changed = False
        for s in list(states):
            # 跳過初始和接受狀態
            if s == dfa.initial_state or s.is_accepting:
                continue
            # 找 in_edges 和 out_edges
            in_edges = [(p, sym) for p in states for sym, q in p.transitions.items() if q == s]
            out_edges = [(sym, q) for sym, q in s.transitions.items()]
            if len(in_edges) == 1 and len(out_edges) == 1:
                p, sym_in = in_edges[0]
                sym_out, q = out_edges[0]
                # 合併 symbol
                merged_symbol = (sym_in, sym_out)
                p.transitions[merged_symbol] = q
                # 刪除舊轉移
                del p.transitions[sym_in]
                del s.transitions[sym_out]
                states.remove(s)
                dfa.states.remove(s)
                changed = True
                break
    return dfa

def merge_parallel_edges(dfa):
    """
    合併 dfa 中平行的 edge
    """
    for s in dfa.states:
        to_syms = {} # (to_state, [symbol1, symbol2, ...])
        for sym, t in list(s.transitions.items()):
            if t not in to_syms:
                to_syms[t] = []
            to_syms[t].append(sym)
        # 若有多條到同一 state，合併
        for t, syms in to_syms.items():
            if len(syms) > 1:
                for sym in syms:
                    del s.transitions[sym]
                s.transitions[tuple(sorted(syms))] = t
    return dfa

