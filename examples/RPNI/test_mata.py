import re

def fix_mata_format(input_file, output_file):
    """
    將 .mata 檔案中的非法狀態名稱 (如 s(s0,s1)) 改成合法的 (s_s0_s1)
    並保持其餘部分不變
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed_lines = []
    pattern = re.compile(r's\([^)]*\)')  # 匹配 s(...) 的狀態名稱

    for line in lines:
        def replace_state(match):
            name = match.group(0)
            # 例如 s(s0,s1) → s_s0_s1
            return name.replace("(", "_").replace(")", "").replace(",", "_")
        fixed_line = pattern.sub(replace_state, line)
        fixed_lines.append(fixed_line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)

    print(f"[OK] 已修正 Mata 檔案格式：{output_file}")

# fix_mata_format("inter_dfa.mata", "inter_dfa_fixed.mata")

import libmata.parser as parser, libmata.alphabets as alph
alpha = alph.OnTheFlyAlphabet()

automata = parser.from_mata("dfa_explicit.mata", alpha)
print("States:", automata.num_of_states())
print("Transitions:", automata.get_num_of_transitions())


