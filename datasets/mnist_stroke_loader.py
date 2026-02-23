# data/mnist_stroke_loader.py

import pickle
import math
from sklearn.model_selection import train_test_split

def compress_stroke_sequence(seq):
    # 將連續相同符號合併，只保留一個
    if not seq:
        return seq
    compressed = [seq[0]]
    for s in seq[1:]:
        if s != compressed[-1]:
            compressed.append(s)
    return compressed

def discretize_4dir(dx, dy):
    """
    4方向離散化
    'R', 'L', 'U', 'D', 'S' (stay)
    """
    if abs(dx) >= abs(dy):  # 水平方向
        if dx > 0:
            return 'R'  # Right
        elif dx < 0:
            return 'L'  # Left
    else:  # 垂直方向
        if dy < 0:
            return 'U'  # Up
        elif dy > 0:
            return 'D'  # Down
    return 'S'  # Stay


def discretize_4dir_pen(dx, dy, pen):
    """
    4方向 + pen狀態 = 8個符號
    pen_down (畫筆落下): 0-4
    pen_up (畫筆抬起):   5-9
    """
    direction = discretize_4dir(dx, dy)
    # pen_down: R, L, U, D, S; pen_up: r, l, u, d, s
    if pen == 1:
        return direction.lower()
    else:
        return direction


def discretize_8dir(dx, dy):
    """
    8方向離散化（英文縮寫）
    'R', 'UR', 'U', 'UL', 'L', 'DL', 'D', 'DR', 'S'
    """
    if dx == 0 and dy == 0:
        return 'S'
    directions = ['R', 'UR', 'U', 'UL', 'L', 'DL', 'D', 'DR']
    angle = math.atan2(dy, dx)
    idx = int((angle + math.pi + math.pi/8) / (math.pi / 4)) % 8
    return directions[idx]


def discretize_8dir_pen(dx, dy, pen):
    """
    8方向 + pen狀態 = 16個符號
    pen_down: 0-7
    pen_up:   8-15
    """
    direction = discretize_8dir(dx, dy)
    # pen_down: E, NE, N, NW, W, SW, S, SE; pen_up: e, ne, n, nw, w, sw, s, se
    if pen == 1:
        return direction.lower()
    else:
        return direction

def _sign(v):
    """Normalize to Python int sign so symbols match the alphabet."""
    if v > 0: return 1
    elif v < 0: return -1
    return 0

def _point_to_symbol(discretize_mode, dx, dy, pen=0):
    """將單一筆劃點離散化為符號"""
    if discretize_mode == "raw_xy":
        return (_sign(dx), _sign(dy))
    elif discretize_mode == "raw":
        return (_sign(dx), _sign(dy), int(pen))
    elif discretize_mode == "4dir":
        return discretize_4dir(dx, dy)
    elif discretize_mode == "4dir_pen":
        return discretize_4dir_pen(dx, dy, pen)
    elif discretize_mode == "8dir":
        return discretize_8dir(dx, dy)
    elif discretize_mode == "8dir_pen":
        return discretize_8dir_pen(dx, dy, pen)
            
def load_mnist_stroke_sequences(
    data_path="datasets/mnist-digits-as-stroke-sequences/mnist_strokes.pkl",
    discretize_mode="raw_xy",  # "4dir", "4dir_pen", "8dir", "8dir_pen", "raw"
    allow_segment=True,
    n_segments=40,
    min_segments=None,
    max_segments=None
):
    """
    Load MNIST stroke sequence data and discretize into symbolic sequences.
    If allow_segment=True, each sequence is divided into segments and each segment is mapped to a symbol (direction).
    The number of segments for each sequence is randomly chosen between min_segments and max_segments (if both are set),
    otherwise n_segments is used for all sequences.
    If allow_segment=False, sequences are not segmented and are discretized pointwise.

    Parameters
    ----------
    data_path : str
        Path to the pickle file containing stroke data.
    discretize_mode : str
        Discretization mode: "4dir", "4dir_pen", "8dir", "8dir_pen", "raw_xy", "raw".
    allow_segment : bool
        Whether to segment sequences into fixed-length blocks.
    n_segments : int
        Default number of segments if min_segments/max_segments are not set.
    min_segments : int or None
        Minimum number of segments (if random segmentation is enabled).
    max_segments : int or None
        Maximum number of segments (if random segmentation is enabled).

    Returns
    -------
    X_train, X_test, y_train, y_test :
        Train/test split of symbolic sequences and labels.
    """

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    X, y = [], []

    def segment_average_direction(seq, discretize_mode="raw_xy", n_segments=10):
        """
        Segment sequence into n_segments blocks and map each block to a symbol (direction).
        For each segment, compute the average (dx, dy) and discretize to a symbol.
        If a segment is empty, inherit previous symbol or use fallback.
        """
        L = len(seq)
        indices = [int(round(i * L / n_segments)) for i in range(n_segments + 1)]
        result_syms = []
        prev_sym = None
        for i in range(n_segments):
            seg = seq[indices[i]:indices[i+1]]
            if not seg:
                # Empty segment: inherit previous symbol or use fallback.
                if prev_sym is not None:
                    result_syms.append(prev_sym)
                else:
                    fallback = _point_to_symbol(discretize_mode, 0, 0, 0)
                    for j in range(i + 1, n_segments):
                        future_seg = seq[indices[j]:indices[j+1]]
                        if future_seg:
                            pt = future_seg[0]
                            fpen = pt[2] if len(pt) == 3 else 0
                            fallback = _point_to_symbol(discretize_mode, pt[0], pt[1], fpen)
                            break
                    result_syms.append(fallback)
                    prev_sym = fallback
            else:
                # Compute average dx, dy for the segment
                dx_sum, dy_sum, pen_sum = 0, 0, 0
                for pt in seg:
                    dx_sum += pt[0]
                    dy_sum += pt[1]
                    pen_sum += pt[2] if len(pt) == 3 else 0
                avg_dx = round(dx_sum / len(seg))
                avg_dy = round(dy_sum / len(seg))
                avg_pen = round(pen_sum / len(seg)) if len(seg[0]) == 3 else 0
                sym = _point_to_symbol(discretize_mode, avg_dx, avg_dy, avg_pen)
                result_syms.append(sym)
                prev_sym = sym
        return result_syms


    import random
    for digit, samples in data.items():
        for seq in samples:
            # seq: [(dx, dy, pen), ...]
            seq = seq[1:]

            # 決定本序列的 n_segments
            seg_len = n_segments
            if min_segments is not None and max_segments is not None:
                seg_len = random.randint(min_segments, max_segments)

            if allow_segment:
                # 分段取平均方向
                seq = segment_average_direction(seq, discretize_mode=discretize_mode, n_segments=seg_len)
            else:
                if discretize_mode == "raw_xy":
                    seq = [(_sign(dx), _sign(dy)) for dx, dy, pen in seq]
                elif discretize_mode == "raw":
                    seq = [(_sign(dx), _sign(dy), pen) for dx, dy, pen in seq]
                elif discretize_mode == "4dir":
                    seq = [discretize_4dir(dx, dy) for dx, dy, pen in seq]
                elif discretize_mode == "4dir_pen":
                    seq = [discretize_4dir_pen(dx, dy, pen) for dx, dy, pen in seq]
                elif discretize_mode == "8dir":
                    seq = [discretize_8dir(dx, dy) for dx, dy, pen in seq]
                elif discretize_mode == "8dir_pen":
                    seq = [discretize_8dir_pen(dx, dy, pen) for dx, dy, pen in seq]

            X.append(seq)
            y.append(digit)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# def get_alphabet(discretize_mode):
#     """
#     根據離散化模式回傳對應的 alphabet
#     """
#     if discretize_mode == "4dir":
#         return [0, 1, 2, 3, 4]  # R, L, U, D, stay
#     elif discretize_mode == "4dir_pen":
#         return list(range(10))  # 0-4 pen_down, 5-9 pen_up
#     elif discretize_mode == "8dir":
#         return list(range(8))   # 8 directions
#     elif discretize_mode == "8dir_pen":
#         return list(range(16))  # 8 directions × 2 pen states
#     else:
#         raise ValueError(f"Unknown mode: {discretize_mode}")

