import argparse
import sys
import os

sys.path.insert(0, './src')
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/interpretera')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external_modules/Explaining-FA')))

# ===== importa local search 模組 =====
from external_modules.interpretera.src.local_serach_synthesis import dra_learner
from external_modules.interpretera.src.local_serach_synthesis import portfolio_runner

# ===== 你自己的 samples（之後可以換成真資料）=====
def load_samples():
    S_pos = [
        [0, -1, 5, 3, 7],
        [0, -2, 4, 1, 6],
    ]
    S_neg = [
        [0, -1, 5, 3, 2],
        [0, -1, 4, 0, 3],
    ]
    return S_pos, S_neg


def main():
    # ===== 載入 samples =====
    S_pos, S_neg = load_samples()

    # ===== 初始化 automaton =====
    learner = dra_learner.DRAlearner(
        num_states=4,
        num_registers=2,
        constants=[0, 1],
        pos_samples=S_pos,
        neg_samples=S_neg,
        golden_pos_samples=[],  # or S_pos if you want to reuse
        auto=None,
        theory="Integer",
        debug=True,
    )

    # ===== 執行 mutation-based local search =====
    # learner.run(max_iterations=50, target_score=0.75)

    # ===== 結果 =====
    best_dra = learner.best_dra
    best_score = learner.best_score
    print("Final score:", best_score)
    # 畫出 automaton 圖
    best_dra.visualize("final_dra")

    learner.test_word([0, -1, 5, 3, 7])  # 應該是接受


if __name__ == "__main__":
    main()
