import random
from typing import List, Tuple


# ============================================================
# OG D1: (ab)+ | (bc)+ | (ca)+
# ============================================================

class OGD1:
    """
    OG D1: (ab)+ | (bc)+ | (ca)+
    只允許完整重複的二元循環（至少一次）
    """

    def check(self, seq: List[str]) -> bool:
        if len(seq) < 2 or len(seq) % 2 != 0:
            return False

        patterns = [('a', 'b'), ('b', 'c'), ('c', 'a')]
        for p in patterns:
            ok = True
            for i in range(0, len(seq), 2):
                if tuple(seq[i:i+2]) != p:
                    ok = False
                    break
            if ok:
                return True
        return False

    def generate_samples(
        self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        pos, neg = [], []
        patterns = [['a', 'b'], ['b', 'c'], ['c', 'a']]
        alphabet = ['a', 'b', 'c']

        # Positive samples
        while len(pos) < num_pos:
            pat = random.choice(patterns)
            k = random.randint(1, max_length // 2)
            seq = pat * k
            if len(seq) <= max_length:
                pos.append(seq)

        # Negative samples
        while len(neg) < num_neg:
            l = random.randint(1, max_length)
            s = [random.choice(alphabet) for _ in range(l)]
            if not self.check(s):
                neg.append(s)

        return pos, neg


# ============================================================
# OG D2: (abc)+
# ============================================================

class OGD2:
    """
    OG D2: (abc)+
    只允許完整重複的三元循環（至少一次）
    """

    def check(self, seq: List[str]) -> bool:
        if len(seq) < 3 or len(seq) % 3 != 0:
            return False

        for i in range(0, len(seq), 3):
            if seq[i:i+3] != ['a', 'b', 'c']:
                return False
        return True

    def generate_samples(
        self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        pos, neg = [], []
        alphabet = ['a', 'b', 'c']

        # Positive samples
        while len(pos) < num_pos:
            k = random.randint(1, max_length // 3)
            seq = ['a', 'b', 'c'] * k
            if len(seq) <= max_length:
                pos.append(seq)

        # Negative samples
        while len(neg) < num_neg:
            l = random.randint(1, max_length)
            s = [random.choice(alphabet) for _ in range(l)]
            if not self.check(s):
                neg.append(s)

        return pos, neg


# ============================================================
# OG D3: a+ | b+ | c+
# ============================================================

class OGD3:
    """
    OG D3: a+ | b+ | c+
    只允許連續重複同一字母
    """

    def check(self, seq: List[str]) -> bool:
        if len(seq) == 0:
            return False
        first = seq[0]
        return all(c == first for c in seq)

    def generate_samples(
        self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        pos, neg = [], []
        alphabet = ['a', 'b', 'c']

        # Positive samples
        while len(pos) < num_pos:
            l = random.randint(1, max_length)
            c = random.choice(alphabet)
            pos.append([c] * l)

        # Negative samples
        while len(neg) < num_neg:
            l = random.randint(2, max_length)
            s = [random.choice(alphabet) for _ in range(l)]
            if not self.check(s):
                neg.append(s)

        return pos, neg


# ============================================================
# OG D4: a | b | c   (sanity check)
# ============================================================

class OGD4:
    """
    OG D4: a | b | c
    只接受單一字母（sanity check language）
    """

    def check(self, seq: List[str]) -> bool:
        return len(seq) == 1 and seq[0] in ['a', 'b', 'c']

    def generate_samples(
        self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        pos, neg = [], []
        alphabet = ['a', 'b', 'c']

        # Positive samples
        while len(pos) < num_pos:
            pos.append([random.choice(alphabet)])

        # Negative samples
        while len(neg) < num_neg:
            l = random.randint(2, max_length)
            s = [random.choice(alphabet) for _ in range(l)]
            neg.append(s)

        return pos, neg


# ============================================================
# OG D5: contains substring "abc"
# ============================================================

class OGD5:
    """
    OG D5: 只要包含子字串 'abc' 即接受
    """

    def check(self, seq: List[str]) -> bool:
        return 'abc' in ''.join(seq)

    def generate_samples(
        self, num_pos: int, num_neg: int, max_length: int
    ) -> Tuple[List[List[str]], List[List[str]]]:

        pos, neg = [], []
        alphabet = ['a', 'b', 'c']

        # Positive samples (包含邊界與長路徑)
        while len(pos) < num_pos:
            l = random.randint(3, max_length)
            s = [random.choice(alphabet) for _ in range(l)]
            idx = random.randint(0, l - 3)
            s[idx:idx + 3] = ['a', 'b', 'c']
            pos.append(s)

        # Negative samples
        while len(neg) < num_neg:
            l = random.randint(1, max_length)
            s = [random.choice(alphabet) for _ in range(l)]
            if not self.check(s):
                neg.append(s)

        return pos, neg


# ============================================================
# Language registry
# ============================================================

OG_LANGUAGES = {
    'OG D1': OGD1,
    'OG D2': OGD2,
    'OG D3': OGD3,
    'OG D4': OGD4,
    'OG D5': OGD5,
}
