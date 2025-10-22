import csv
from itertools import product
import re

max_length = 6
cols = [f"s{i}" for i in range(max_length)]
pat = re.compile(r"^(ab)*$") 

with open("ab_tabular.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols + ["label"])
    for length in range(max_length + 1):
        for bits in product("ab", repeat = length):
            seq = "".join(bits)
            y = 1 if pat.fullmatch(seq) else 0
            row = list(seq) + ["_"] * (max_length - length)
            w.writerow(row + [y])