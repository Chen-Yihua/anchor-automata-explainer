import numpy as np
import pandas as pd
from itertools import product

try:
    from robot_operation import robot_instance
    HAS_ROBOT = True
except Exception:
    HAS_ROBOT = False

rng = np.random.default_rng(42)

# N = 243            # 生成筆數
L = 15             # 最大長度
FIXED_LENGTH = False  # True: 固定長度, False: 隨機長度
FILE_NAME = 'robot_randomlen.csv'  # 輸出檔案名稱
COLORS = ['green', 'yellow', 'blue']

def fallback_rule(seq):
    """
    - 至少有一個 'green' 出現在第一個 'blue' 之前，且
    - 序列中包含 'yellow'
    """
    valid = True # 預設為 True

    # 轉成 list of str（確保每個元素都是字串）
    if isinstance(seq, np.ndarray):
        seq = seq.tolist()
    seq = [str(s) for s in seq]  # 防止 float/int

    # 檢查規則
    if 'yellow' not in seq:  
        valid = False  # 必須有 yellow

    # 檢查是否有 blue → yellow，且中間沒 green
    for i in range(1, len(seq)):
        if seq[i] == 'yellow':
            # 向前找最近的 blue
            for j in range(i - 1, -1, -1):
                if seq[j] == 'green':
                    break  # 中間有 green，符合規則
                if seq[j] == 'blue':
                    # 中間沒 green，違反規則
                    valid = False
                    break
        if not valid:
            break

    return int(valid)

rows, labels = [], []
if FIXED_LENGTH:
    # 固定長度：遍歷所有長度為 L 的組合
    for row in product(COLORS, repeat=L):
        row = list(row)
        rows.append(row)
        if HAS_ROBOT:
            labels.append(int(robot_instance.is_valid_path(row)))
        else:
            labels.append(fallback_rule(row))
else:
    # 非固定長度：遍歷所有長度 1~L 的組合
    for length in range(1, L + 1):
        for row in product(COLORS, repeat=length):
            row = list(row)
            rows.append(row)
            if HAS_ROBOT:
                labels.append(int(robot_instance.is_valid_path(row)))
            else:
                labels.append(fallback_rule(row))


cols = [f's{i}' for i in range(L)]
df = pd.DataFrame(rows, columns=cols)
df['label'] = labels
df.to_csv(FILE_NAME, index=False, encoding='utf-8')
