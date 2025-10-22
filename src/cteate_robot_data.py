# synth_robot_csv.py
import numpy as np
import pandas as pd

try:
    from robot_operation import robot_instance
    HAS_ROBOT = True
except Exception:
    HAS_ROBOT = False

rng = np.random.default_rng(42)

N = 243 # 生成筆數
L = 5  # 每筆長度
COLORS = ['green', 'yellow', 'blue']

def rand_row():
    return rng.choice(COLORS, size=L).tolist()

def fallback_rule(path):
    """
    - 至少有一個 'green' 出現在第一個 'blue' 之前，且
    - 序列中包含 'yellow'
    """
    try:
        first_blue = path.index('blue')
    except ValueError:
        first_blue = len(path)
    has_green_before_blue = any(c == 'green' for c in path[:first_blue])
    has_yellow = 'yellow' in path
    return int(has_green_before_blue and has_yellow)

rows, labels = [], []
for _ in range(N):
    row = rand_row()
    rows.append(row)
    if HAS_ROBOT:
        labels.append(int(robot_instance.is_valid_path(row)))
    else:
        labels.append(fallback_rule(row))

cols = [f's{i}' for i in range(L)]
df = pd.DataFrame(rows, columns=cols)
df['label'] = labels
df.to_csv('robot.csv', index=False, encoding='utf-8')
print('Saved robot.csv with shape:', df.shape)
print(df.head())
