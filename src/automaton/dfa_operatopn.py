from typing import Tuple, Dict
from aalpy.automata.Dfa import Dfa, DfaState

def dfa_product(dfa1: Dfa, dfa2: Dfa, final_func) -> Dfa:
    """
    Generic DFA product construction (for intersection/union).
    final_func: function that takes (accept1, accept2) -> bool (accepting)
    """
    # alphabet = list(set(dfa1.get_input_alphabet()) | set(dfa2.get_input_alphabet()))
    alphabet = list(_alphabet_of(dfa1) | _alphabet_of(dfa2))

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


def get_base_dfa(learn_type, alphabet_map, features, test_instance) -> Dfa:
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

        if learn_type != 'Tabular':
            # if step in features:
                # transitions = {f'p_({i},1.0)': next_state}
            # else:
            transitions = {f'p{i}_({symbol})': next_state for symbol in alphabet_map[step]}
        else:
            # if step in features:
                # transitions = {test_instance[i]: next_state}
            # else:
            transitions = {symbol: next_state for symbol in alphabet_map[step]}
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
    states = set(dfa.states)
    changed = True
    while changed:
        changed = False
        for s in list(states):
            if s is dfa.initial_state or s.is_accepting:
                continue

            # 找 s 的唯一入邊與唯一出邊
            in_edges = []
            for p in states:
                for sym, q in p.transitions.items():
                    if q is s:
                        in_edges.append((p, sym)) # 點，邊
                        if len(in_edges) > 1:
                            break
                if len(in_edges) > 1:
                    break

            out_edges = list(s.transitions.items()) # 邊，點
            if len(in_edges) == 1 and len(out_edges) == 1:
                p, sym_in = in_edges[0] 
                sym_out, q = out_edges[0]

                # 標籤相同時才合併，避免產生 tuple
                if sym_in == sym_out:
                    # 將 p 指向 q
                    p.transitions[sym_in] = q
                    # 刪掉 s（不需要先刪 s→q，那會在移除 s 後自然消失）
                    states.remove(s)
                    dfa.states.remove(s)
                    changed = True
                    break
    return dfa

def merge_parallel_edges(dfa, learn_type):
    """
    合併 dfa 中平行的 edge，若 type 是 'Text' 或 'Image'，會先移除位置資訊 (p0_(x.x) -> x.x)
    """   
    # import re
    # # 清理帶位置的邊，如 p0_(1.0) -> 1.0
    # def clean_symbol(sym: str) -> str:
    #     if learn_type == "Text" or learn_type == "Image":
    #         match = re.search(r"\(([\d\.\-]+)\)", sym)
    #         if match:
    #             return match.group(1) 
    #         return sym
    #     return sym

    # alphabet = set(clean_symbol(sym) for sym in _alphabet_of(dfa))

    # for s in dfa.states:
    #     target_map = {} # 狀態: 邊
    #     new_transitions = {}
    #     for sym, t in s.transitions.items():
    #         clean_sym = clean_symbol(sym)
    #         new_transitions[clean_sym] = t
    #         if t != s: # 忽略 self loop
    #             target_map.setdefault(t, set()).add(clean_sym)

    #     # 若有平行邊涵蓋了所有 alphabet，改成 *
    #     for t, symbols in target_map.items():
    #         if symbols == alphabet:
    #             s.transitions = new_transitions
    #             for sym in list(symbols):
    #                 if sym in s.transitions and s.transitions[sym] is t:
    #                     del s.transitions[sym]
    #             s.transitions['*'] = t  
    # return dfa
    import re

    def clean_symbol(sym: str) -> str:
        """清理符號的括號資訊"""
        if learn_type in ("Text", "Image"):
            match = re.search(r"\(([\d\.\-]+)\)", sym)
            if match:
                return match.group(1)
            return sym
        return sym

    def get_prefix(sym):
        """取得位置前綴（如 'p1_'）"""
        if not isinstance(sym, str):
            return ""
        match = re.match(r"(p\d+)_\([\d\.\-]+\)", sym)
        return match.group(1) + "_" if match else ""

    alphabet = set(clean_symbol(sym) for sym in _alphabet_of(dfa))

    for s in dfa.states:
        target_map = {}      # t -> set(symbols)
        new_transitions = {}
        prefix_map = {}      # t -> prefix for naming

        # 收集轉移資訊
        for sym, t in s.transitions.items():
            clean_sym = clean_symbol(sym)
            new_transitions[clean_sym] = t
            if t != s:  # 忽略 self-loop
                target_map.setdefault(t, set()).add(clean_sym)
                prefix = get_prefix(sym)
                if prefix:
                    prefix_map[t] = prefix  # 儲存對應的 pX_ 前綴

        # 若有平行邊涵蓋所有 alphabet，合併成 pX_* 或 *
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

def simplify_dfa(dfa, learn_type) :
    """
    簡化 dfa
    """
    dfa = merge_parallel_edges(dfa, learn_type)
    dfa = merge_linear_edges(dfa)
    return dfa

from aalpy.automata.Dfa import Dfa as AalpyDfa, DfaState
def scar_to_aalpy_dfa(scar_dfa) -> AalpyDfa:
    """
    將 scar_rpni_size_capped_demo.DFA 轉成 aalpy.automata.Dfa
    需求：scar_dfa 有 .alphabet, .start, .delta, .accepting, .states
    """
    # 建立對應的狀態
    id2state = {}
    for q in sorted(scar_dfa.states):
        id2state[q] = DfaState(f"s{q}", q in scar_dfa.accepting)

    initial = id2state[scar_dfa.start]

    # 建立轉移（對 scar 的 (q,a)->r 逐一掛到 aalpy state）
    for (q, a), r in scar_dfa.delta.items():
        id2state[q].transitions[a] = id2state[r]

    # 回傳 aalpy 的 Dfa
    return AalpyDfa(initial, list(id2state.values()))

def _alphabet_of(dfa):
    # scar 版
    if hasattr(dfa, "alphabet"):
        return set(dfa.alphabet)
    # aalpy 版（新舊版本有 get_input_alphabet() 或從轉移蒐集）
    if hasattr(dfa, "get_input_alphabet"):
        try:
            return set(dfa.get_input_alphabet())
        except Exception:
            pass
    # fallback：從所有狀態轉移收集
    syms = set()
    for s in getattr(dfa, "states", []):
        syms.update(getattr(s, "transitions", {}).keys())
    return syms

def dfa_intersection_any(d1, d2):
    # 若不是 aalpy.Dfa，就先轉
    if not isinstance(d1, AalpyDfa):
        d1 = scar_to_aalpy_dfa(d1)
    if not isinstance(d2, AalpyDfa):
        d2 = scar_to_aalpy_dfa(d2)
    return dfa_intersection(d1, d2)

