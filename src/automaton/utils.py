def get_alphabet_from_samples(pos, neg):
    """
    從樣本中取得字母表
    : param pos: List[List[str]]，正例樣本
    : param neg: List[List[str]]，負例樣本
    : return: List[str]，字母表
    """
    sigma = set()
    for S in (pos, neg):
        for x in S:
            sigma.update(x)
    return sorted(sigma)

def add_position_to_sample(samples):
    """
    將每個樣本特徵加上位置資訊
    """
    return [[f'p{i}_({v})' for i, v in enumerate(sample)] for sample in samples]

def append_eos(batch, eos="EOS"):
    out = []
    for seq in batch:
        s = list(seq)
        if not s or s[-1] != eos: # 避免重複補（若外部已補過）
            s.append(eos)
        out.append(s)
    return out

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
    # current_state = dfa.start
    # for a in path:
    #     current_state = dfa.delta.get((current_state, a), dfa.sink)
    #     if current_state == dfa.sink:
    #         return False
    # return True

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
    # current_state = dfa.start
    # for a in path:
    #     current_state = dfa.delta.get((current_state, a), dfa.sink)
    #     if current_state == dfa.sink:
    #         return False
    # return current_state in dfa.accepting

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

def dfa_to_graphviz(dfa, show_sink=False):
    lines = []
    lines.append("digraph DFA {")
    lines.append('  rankdir=LR;')
    lines.append('  node [shape=circle];')

    # 起始點
    lines.append('  __start__ [shape=point];')
    lines.append(f'  __start__ -> "{dfa.start}";')

    # 接受節點
    for q in sorted(dfa.states):
        if not show_sink and hasattr(dfa, "sink") and q == dfa.sink:
            continue
        shape = "doublecircle" if q in dfa.accepting else "circle"
        lines.append(f'  "{q}" [shape={shape}];')

    # 邊
    for (q, a), r in sorted(dfa.delta.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if not show_sink and hasattr(dfa, "sink") and (q == dfa.sink or r == dfa.sink):
            continue
        lines.append(f'  "{q}" -> "{r}" [label="{a}"];')

    lines.append("}")
    return "\n".join(lines)

# def plot_precision_vs_dfa_size(results):
#     import matplotlib.pyplot as plt
#     state_list = [states for _, states, _ in results]
#     precs = [precision for _, _, precision in results]
#     plt.figure()
#     plt.plot(state_list, precs, marker='o')
#     plt.xlabel("Number of states")
#     plt.ylabel("Precision")
#     plt.title("SCAR-RPNI: Precision vs. DFA size")
#     plt.grid(True)
#     for x, y in zip(state_list, precs):
#         plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
#     plt.show()


import json
def dfa_to_mata(dfa, file_path):
    """
    匯出符合 libMata 1.23.3 (C++ parser) 格式的 @NFA-explicit 檔案
    - 所有狀態以 q 開頭
    - 所有符號以整數 ID 表示
    - 輸出 Mata 檔，回傳 state_map.json、symbol_map.json (對應表)
    """

    # 狀態名稱轉換
    def clean_state_name(name):
        name = str(name)
        name = ''.join(ch for ch in name if ch.isalnum())
        if not name.startswith("q"):
            name = "q" + name
        return name

    # 取出 DFA 結構
    states = list(dfa.states)
    init_state = dfa.initial_state
    finals = [s for s in states if s.is_accepting]

    # 建立對應表
    all_syms = sorted({sym for s in states for sym in s.transitions.keys()})
    symbol_map = {sym: i for i, sym in enumerate(all_syms)}
    id_to_sym = {i: sym for sym, i in symbol_map.items()}
    state_map = {s.state_id: clean_state_name(s.state_id) for s in states}

    # 寫入 Mata 檔
    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("@NFA-explicit\n")
        f.write(f"%Initial {state_map[init_state.state_id]}\n")

        if finals:
            finals_str = " ".join(state_map[s.state_id] for s in finals)
            f.write(f"%Final {finals_str}\n")

        for s in states:
            src = state_map[s.state_id]
            for sym, tgt in s.transitions.items():
                tgt_name = state_map[tgt.state_id]
                sym_id = symbol_map[sym]
                f.write(f"{src} {sym_id} {tgt_name}\n")

    # 儲存 mapping JSON
    # json_path = file_path.replace(".mata", "_map.json")
    # with open(json_path, "w", encoding="utf-8") as jf:
    #     json.dump({
    #         "state_map": state_map,          # 例如 {"s0s0": "q0", "s0s1": "q1", ...}
    #         "symbol_map": symbol_map,         # 例如 {"p0_(1.0)": 0, "p1_(1.0)": 1, ...}
    #         "inverse_symbol_map": id_to_sym  # 例如 {0: "p0_(1.0)", 1: "p1_(1.0)", ...}
    #     }, jf, indent=2, ensure_ascii=False)

    # print(f"state_map = {state_map}\n")
    # print(f"symbol_map = {symbol_map}\n")
    return state_map, symbol_map

def get_test_word(mata_path, symbol_map):
    import re
    from collections import defaultdict
    grouped = defaultdict(list)
    path = []

    # 先按位置分組，例如 p1_(1.0), p1_(0.0) → group['p1']
    for k, v in symbol_map.items():
        match = re.match(r"(p\d+)_", k)
        if not match:
            continue
        pos = match.group(1)
        grouped[pos].append((k, v))

    # 每個位置只走一次
    for pos in sorted(grouped.keys(), key=lambda x: int(x[1:])):  # 根據數字排序
        symbols = grouped[pos]
        for k, v in symbols:
            if "(1.0)" in k or "*" in k:
                path.append((pos, v))
                break  # 一個位置走完就換下一個

    # 取出 symbol id
    path_ids = [v for _, v in path]
    print("路徑:", path_ids)
    return path_ids
    # import re
    # from collections import defaultdict

    # # === 讀取 MATA 檔 ===
    # with open(mata_path, encoding="utf-8") as f:
    #     lines = [l.strip() for l in f if l.strip()]

    # transitions = defaultdict(list)
    # start = None
    # finals = set()

    # for line in lines:
    #     if line.startswith('%Initial'):
    #         start = line.split()[1]
    #     elif line.startswith('%Final'):
    #         finals.add(line.split()[1])
    #     elif not line.startswith('@') and not line.startswith('%'):
    #         src, sym, dst = line.split()
    #         transitions[src].append((int(sym), dst))

    # # 允許的 symbol id（1.0 或 *
    # valid_syms = {v for k, v in symbol_map.items() if "(1.0)" in k or "*" in k}

    # cur = start
    # visited = set()
    # path_syms = []

    # while True:
    #     visited.add(cur)
    #     found = False
    #     for sym, dst in transitions[cur]:
    #         if sym in valid_syms and dst not in visited:
    #             path_syms.append(sym)
    #             cur = dst
    #             found = True
    #             break
    #     if not found or cur in finals:
    #         break
    # print("路徑:", path_syms)
    # return path_syms

def explain_axp_cxp(axps, cxps, symbol_map):
    inv_map = {v: k for k, v in symbol_map.items()}
    for i, axp in enumerate(axps):
        print(f"AXp {i+1}: {[inv_map[x] for x in axp]}")
    for i, cxp in enumerate(cxps):
        print(f"Cxp {i+1}: {[inv_map[x] for x in cxp]}")