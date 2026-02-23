import numpy as np
import random
import argparse
import sys
from typing import List, Tuple, Optional

# ==========================================
# 1. 核心基類 (解決偏差與效率問題)
# ==========================================

class Language:
    """
    Improved Base Class for Automata Learning Benchmarks.
    Features:
    1. Constructive Positive Sampling (Efficient)
    2. Mutational Negative Sampling (Robust, Unbiased)
    3. Dynamic Parameter Ranges (Prevents Overfitting)
    """
    def __init__(self, num_pos=0, num_neg=0, max_len=20):
        # 關鍵修正：縮小值範圍，減少擾動時的噪音
        # 原本 [-100, 100] 太大，導致擾動後的樣本難以區分
        self.val_min = 0      # 使用非負整數，簡化比較
        self.val_max = 50     # 較小範圍，提高學習效率
        self.min_len = 3      # 縮短最小長度，允許更多變化
        self.step_range = (1, 10) # 較小的步長範圍
        
    def check(self, sequence: List[float]) -> bool:
        """Returns True if sequence belongs to the language."""
        raise NotImplementedError

    def generate_positive(self, max_length: int) -> List[float]:
        """Generates a guaranteed valid sequence."""
        raise NotImplementedError

    def mutate_sequence(self, sequence: List[float], strategy: str = 'mixed') -> List[float]:
        """
        Core Logic for generating unbiased negative samples.
        Takes a valid sequence and breaks it slightly at a RANDOM location.
        """
        if len(sequence) < 2:
            # Re-generate a random singleton that is likely invalid for most complex rules
            return [float(np.random.randint(self.val_min, self.val_max))]
        
        new_seq = sequence.copy()
        # CRITICAL FIX: Mutate at a random index to remove Location Bias
        idx = random.randint(0, len(new_seq) - 1)
        
        if strategy == 'mixed':
            strategies = ['noise', 'swap', 'replace', 'length']
            strategy = random.choice(strategies)

        if strategy == 'noise':
            # Add significant noise to break numerical relationships (e.g. increasing)
            noise = np.random.choice([-1, 1]) * np.random.randint(10, 50)
            new_seq[idx] += noise
            
        elif strategy == 'swap':
            # Swap two elements (breaks ordering)
            idx2 = random.randint(0, len(new_seq) - 1)
            new_seq[idx], new_seq[idx2] = new_seq[idx2], new_seq[idx]
            
        elif strategy == 'replace':
            # Replace with a completely random value (breaks pattern)
            new_val = float(np.random.randint(self.val_min, self.val_max))
            # Ensure we actually changed the value
            attempts = 0
            while new_val == new_seq[idx] and attempts < 5:
                new_val = float(np.random.randint(self.val_min, self.val_max))
                attempts += 1
            new_seq[idx] = new_val
            
        elif strategy == 'length':
            # Insert or Delete (breaks structural/modulo rules)
            if len(new_seq) > self.min_len and random.random() > 0.5:
                new_seq.pop(idx)
            else:
                new_seq.insert(idx, float(np.random.randint(self.val_min, self.val_max)))
                
        return new_seq

    def generate_samples(self, num_pos: int, num_neg: int, max_length: int) -> Tuple[List[List[float]], List[List[float]]]:
        pos_sample = []
        neg_sample = []

        # 1. Generate Positives (Constructive)
        attempts = 0
        while len(pos_sample) < num_pos and attempts < num_pos * 10:
            seq = self.generate_positive(max_length)
            if self.check(seq): 
                pos_sample.append(seq)
            attempts += 1

        # 2. Generate Negatives via Mutation (Near-Miss)
        attempts = 0
        while len(neg_sample) < num_neg and attempts < num_neg * 100:
            # Pick a positive sample to corrupt (ensures similar length/distribution)
            if pos_sample:
                base_seq = random.choice(pos_sample)
            else:
                base_seq = self.generate_positive(max_length)
            
            mutated_seq = self.mutate_sequence(base_seq)
            
            # Critical: Verify it is actually negative
            if not self.check(mutated_seq):
                neg_sample.append(mutated_seq)
            
            attempts += 1
                
        return pos_sample, neg_sample

# ==========================================
# 2. S-Series: Numerical Patterns
# ==========================================

class S1(Language):
    """Strictly Increasing"""
    def check(self, arr: List[float]) -> bool:
        if len(arr) < 2: return True
        return all(arr[i] > arr[i-1] for i in range(1, len(arr)))

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.min_len, max_length + 1)
        # 確保有足夠空間讓序列遞增
        max_start = self.val_max - length * self.step_range[1]
        if max_start <= self.val_min:
            max_start = self.val_min + 10  # 安全下界
        start = np.random.randint(self.val_min, max(self.val_min + 1, max_start))
        seq = [float(start)]
        curr = start
        for _ in range(length - 1):
            curr += np.random.randint(self.step_range[0], self.step_range[1])
            seq.append(float(curr))
        return seq

class S2(Language):
    """Strictly Decreasing"""
    def check(self, arr: List[float]) -> bool:
        if len(arr) < 2: return True
        return all(arr[i] < arr[i-1] for i in range(1, len(arr)))

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.min_len, max_length + 1)
        # 從較大的值開始，確保有足夠空間遞減
        start = np.random.randint(self.val_max - 10, self.val_max)
        seq = [float(start)]
        curr = start
        for _ in range(length - 1):
            step = np.random.randint(self.step_range[0], self.step_range[1])
            curr = curr - step
            seq.append(float(curr))
        return seq

class S3(Language):
    """Weakly Increasing"""
    def check(self, arr: List[float]) -> bool:
        if len(arr) < 2: return True
        return all(arr[i] >= arr[i-1] for i in range(1, len(arr)))

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.min_len, max_length + 1)
        # 確保有足夠空間讓序列弱遞增
        max_start = self.val_max - length * self.step_range[1]
        if max_start <= self.val_min:
            max_start = self.val_min + 10
        start = np.random.randint(self.val_min, max(self.val_min + 1, max_start))
        seq = [float(start)]
        curr = start
        for _ in range(length - 1):
            curr += np.random.randint(0, self.step_range[1]) # Allow 0 step
            seq.append(float(curr))
        return seq

class S4(Language):
    """Weakly Decreasing"""
    def check(self, arr: List[float]) -> bool:
        if len(arr) < 2: return True
        return all(arr[i] <= arr[i-1] for i in range(1, len(arr)))

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.min_len, max_length + 1)
        # 確保有足夠空間讓序列弱遞減
        min_start = self.val_min + length * self.step_range[1]
        if min_start >= self.val_max:
            min_start = self.val_max - 10
        start = np.random.randint(max(self.val_min, min_start), self.val_max)
        seq = [float(start)]
        curr = start
        for _ in range(length - 1):
            curr -= np.random.randint(0, self.step_range[1])
            seq.append(float(curr))
        return seq
        return seq

class S5(Language):
    """Up then Down (Peak)"""
    def check(self, lst: List[float]) -> bool:
        if len(lst) < 3: return False
        lst = list(lst)
        peak_val = max(lst)
        
        # Handle multiple peaks: if any valid peak exists, return True
        for peak_idx in [i for i, v in enumerate(lst) if v == peak_val]:
            if peak_idx == 0 or peak_idx == len(lst) - 1:
                continue

            is_valid_peak = True
            # Check Up
            for i in range(1, peak_idx + 1):
                if lst[i] <= lst[i-1]:
                    is_valid_peak = False
                    break
            if not is_valid_peak:
                continue

            # Check Down
            for i in range(peak_idx + 1, len(lst)):
                if lst[i] >= lst[i-1]:
                    is_valid_peak = False
                    break
            
            if is_valid_peak:
                return True # Found a valid peak structure
                
        return False

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(5, max_length + 1)
        peak_idx = np.random.randint(1, length - 1)
        
        # Generate Up part - 確保有足夠空間
        max_start = self.val_max - length * 20  # 使用實際步長上界
        if max_start <= self.val_min:
            max_start = self.val_min + 1
        start = np.random.randint(self.val_min, max(self.val_min + 1, max_start))
        seq = [float(start)]
        curr = start
        for _ in range(peak_idx):
            curr += np.random.randint(1, 20)
            seq.append(float(curr))
            
        # Generate Down part
        for _ in range(length - 1 - peak_idx):
            curr -= np.random.randint(1, 20)
            seq.append(float(curr))
        return seq

class S6(Language):
    """Down then Up (Valley)"""
    def check(self, lst: List[float]) -> bool:
        if len(lst) < 3: return False
        lst = list(lst)
        min_val = min(lst)

        # Handle multiple valleys
        for min_idx in [i for i, v in enumerate(lst) if v == min_val]:
            if min_idx == 0 or min_idx == len(lst) - 1:
                continue

            is_valid_valley = True
            # Check Down
            for i in range(1, min_idx + 1):
                if lst[i] >= lst[i-1]:
                    is_valid_valley = False
                    break
            if not is_valid_valley:
                continue

            # Check Up
            for i in range(min_idx + 1, len(lst)):
                if lst[i] <= lst[i-1]:
                    is_valid_valley = False
                    break
            
            if is_valid_valley:
                return True
                
        return False

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(5, max_length + 1)
        min_idx = np.random.randint(1, length - 1)
        
        # 確保有足夠空間讓序列先下降再上升
        min_start = self.val_min + length * 20  # 需要足夠高才能下降
        if min_start >= self.val_max:
            min_start = self.val_max - 1
        start = np.random.randint(max(self.val_min, min_start), self.val_max)
        seq = [float(start)]
        curr = start
        for _ in range(min_idx):
            curr -= np.random.randint(1, 20)
            seq.append(float(curr))
        for _ in range(length - 1 - min_idx):
            curr += np.random.randint(1, 20)
            seq.append(float(curr))
        return seq

class AlternatingPrefix(Language):
    """Helper for S7, S8, S12, S13, S14 (Up/Down patterns)"""
    def __init__(self, pattern_len=3):
        super().__init__()
        self.pattern_len = pattern_len # Number of transitions to check
        # Pattern: True=Up, False=Down. Alternating starting with Up.
        self.pattern = [True if i % 2 == 0 else False for i in range(pattern_len)]

    def check(self, lst: List[float]) -> bool:
        if len(lst) < self.pattern_len + 1: return False
        diffs = [lst[i] - lst[i-1] for i in range(1, len(lst))]
        
        # Check prefix pattern
        for i in range(self.pattern_len):
            if self.pattern[i]: # Expect Up
                if diffs[i] <= 0: return False
            else: # Expect Down
                if diffs[i] >= 0: return False
        return True

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.pattern_len + 2, max_length + 1)
        curr = float(np.random.randint(0, 50))
        seq = [curr]
        
        # Generate Prefix
        for is_up in self.pattern:
            if is_up:
                curr += np.random.randint(5, 20)
            else:
                curr -= np.random.randint(5, 20)
            seq.append(curr)
            
        # Fill rest randomly (but validly? Original code allowed anything after prefix)
        # To avoid accidental violation of a "global" rule if there were one, but here only prefix matters.
        for _ in range(length - len(seq)):
            curr += np.random.randint(-10, 11) # Random walk
            seq.append(curr)
        return seq

# Specific Alternating Classes
class S7(AlternatingPrefix): 
    def __init__(self): super().__init__(pattern_len=3) # Up Down Up
class S8(AlternatingPrefix): 
    def __init__(self): super().__init__(pattern_len=4)
class S12(AlternatingPrefix): 
    def __init__(self): super().__init__(pattern_len=5)
class S13(AlternatingPrefix): 
    def __init__(self): super().__init__(pattern_len=6)
class S14(AlternatingPrefix): 
    def __init__(self): super().__init__(pattern_len=7)

class PeaksTroughs(Language):
    """Base for S9, S10, S11"""
    def __init__(self, peak_rule='inc', trough_rule='inc'):
        super().__init__()
        self.peak_rule = peak_rule # 'inc' or 'dec'
        self.trough_rule = trough_rule

    def get_peaks_troughs(self, lst):
        peaks, troughs = [], []
        if len(lst) < 3: return peaks, troughs # Simplification
        
        # Identify local extrema
        # Note: Original code handles endpoints specifically. Let's match general logic.
        # We will scan internal points.
        for i in range(1, len(lst)-1):
            if lst[i] > lst[i-1] and lst[i] > lst[i+1]: peaks.append(lst[i])
            if lst[i] < lst[i-1] and lst[i] < lst[i+1]: troughs.append(lst[i])
            
        # Handle endpoints like original code (simplified for robustness)
        if len(lst) >= 2:
            if lst[-1] > lst[-2]: peaks.append(lst[-1])
            elif lst[-1] < lst[-2]: troughs.append(lst[-1])
            
        return peaks, troughs

    def check(self, lst: List[float]) -> bool:
        if len(lst) < 2: return True # Trivial
        peaks, troughs = self.get_peaks_troughs(lst)
        
        # Check Peaks
        if self.peak_rule == 'inc':
            for i in range(1, len(peaks)): 
                if peaks[i] <= peaks[i-1]: return False
        elif self.peak_rule == 'dec':
            for i in range(1, len(peaks)): 
                if peaks[i] >= peaks[i-1]: return False
                
        # Check Troughs
        if self.trough_rule == 'inc':
            for i in range(1, len(troughs)): 
                if troughs[i] <= troughs[i-1]: return False
        elif self.trough_rule == 'dec':
            for i in range(1, len(troughs)): 
                if troughs[i] >= troughs[i-1]: return False
                
        return True

    def generate_positive(self, max_length: int) -> List[float]:
        # Construct zigzag with controlling envelopes
        length = np.random.randint(5, max_length + 1)
        seq = []
        
        # Envelopes
        peak_env = 100.0
        trough_env = 0.0
        
        for i in range(length):
            # Toggle between high and low to create peaks/troughs
            if i % 2 == 0: # Peak Candidate
                val = peak_env
                seq.append(val)
                # Update envelope
                if self.peak_rule == 'inc': peak_env += np.random.randint(5, 15)
                else: peak_env -= np.random.randint(5, 15)
            else: # Trough Candidate
                val = trough_env
                seq.append(val)
                if self.trough_rule == 'inc': trough_env += np.random.randint(5, 15)
                else: trough_env -= np.random.randint(5, 15)
                
        return seq

class S9(PeaksTroughs): # Higher Peaks, Higher Troughs
    def __init__(self): super().__init__('inc', 'inc')
class S10(PeaksTroughs): # Higher Peaks, Lower Troughs
    def __init__(self): super().__init__('inc', 'dec')
class S11(PeaksTroughs): # Lower Peaks, Lower Troughs
    def __init__(self): super().__init__('dec', 'dec')

# ==========================================
# 3. L-Series: Structural Patterns
# ==========================================

class L1(Language):
    """Constant Sequence (a, a, a...)"""
    def check(self, lst: List[float]) -> bool:
        if len(lst) == 0: return True
        return all(x == lst[0] for x in lst)

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.min_len, max_length + 1)
        val = float(np.random.randint(self.val_min, self.val_max))
        return [val] * length

class RepeatingPattern(Language):
    """Generic class for (ab)*, (abc)*, (aaaa)* etc."""
    def __init__(self, pattern_len=2, distinct_count=2):
        super().__init__()
        self.pat_len = pattern_len
        self.distinct = distinct_count # 1 for (a)*, 2 for (ab)*

    def check(self, lst: List[float]) -> bool:
        if len(lst) == 0: return True
        lst = list(lst)  # Convert numpy array to list
        if len(set(lst)) != self.distinct: return False
        if len(lst) % self.pat_len != 0: return False
        
        # Check repetition
        block = lst[:self.pat_len]
        for i in range(0, len(lst), self.pat_len):
            if lst[i:i+self.pat_len] != block: return False
        return True

    def generate_positive(self, max_length: int) -> List[float]:
        repeats = max(1, np.random.randint(1, max_length // self.pat_len + 1))
        
        # Generate block with exactly self.distinct unique values
        if self.distinct == 1:
            # All elements in block are the same
            val = float(np.random.randint(0, 20))
            block = [val] * self.pat_len
        else:
            # Need exactly self.distinct unique values in the block
            block = []
            attempts = 0
            while len(set(block)) != self.distinct and attempts < 1000:
                # Generate block ensuring we have exactly distinct unique values
                values = [float(np.random.randint(0, 20)) for _ in range(self.distinct)]
                # Make sure values are actually distinct
                while len(set(values)) != self.distinct:
                    values = [float(np.random.randint(0, 20)) for _ in range(self.distinct)]
                # Fill block by cycling through values
                block = []
                for i in range(self.pat_len):
                    block.append(values[i % self.distinct])
                attempts += 1
            
        return block * repeats

# Specific Structural Classes
class L2(RepeatingPattern): # (ab)*
    def __init__(self): super().__init__(2, 2)
class L2A(RepeatingPattern): # (aa)* -> effectively (a)* even length
    def __init__(self): super().__init__(2, 1)
class L3A(RepeatingPattern): # (aaa)* 
    def __init__(self): super().__init__(3, 1)
class L4A(RepeatingPattern): 
    def __init__(self): super().__init__(4, 1)
class L5A(RepeatingPattern): 
    def __init__(self): super().__init__(5, 1)
class L6A(RepeatingPattern): 
    def __init__(self): super().__init__(6, 1)
class L7A(RepeatingPattern): 
    def __init__(self): super().__init__(7, 1)
class L8A(RepeatingPattern): 
    def __init__(self): super().__init__(8, 1)
class L9A(RepeatingPattern): 
    def __init__(self): super().__init__(9, 1)
class L10A(RepeatingPattern): 
    def __init__(self): super().__init__(10, 1)

class L2AB(RepeatingPattern): # (abab)* -> length mod 4, 2 distinct
    def __init__(self): super().__init__(4, 2)
class L3AB(RepeatingPattern): # (ababab)* -> length mod 6
    def __init__(self): super().__init__(6, 2)
class L4AB(RepeatingPattern): 
    def __init__(self): super().__init__(8, 2)
class L5AB(RepeatingPattern): 
    def __init__(self): super().__init__(10, 2)

class L3(Language):
    """Parity of counts: odd A, even B. Max 2 symbols."""
    def check(self, lst: List[float]) -> bool:
        if len(lst) == 0: return False
        lst = list(lst)  # Convert numpy array to list
        unique = list(set(lst))
        if len(unique) > 2: return False
        
        a = unique[0]
        count_a = lst.count(a)
        if len(unique) == 1:
            return count_a % 2 == 1 # Unary: odd length
            
        b = unique[1]
        count_b = lst.count(b)
        
        # Check rule: (Odd A, Even B) OR (Odd B, Even A)
        c1 = (count_a % 2 == 1 and count_b % 2 == 0)
        c2 = (count_b % 2 == 1 and count_a % 2 == 0)
        return c1 or c2

    def generate_positive(self, max_length: int) -> List[float]:
        a = float(np.random.randint(0, 10))
        b = float(np.random.randint(11, 20))
        
        count_a = 2 * np.random.randint(1, 5) + 1 # Odd
        count_b = 2 * np.random.randint(1, 5)     # Even
        
        seq = [a]*count_a + [b]*count_b
        random.shuffle(seq)
        return seq

class L4(Language):
    """No AAA (No 3 consecutive identical)"""
    def check(self, sequence: List[float]) -> bool:
        for i in range(len(sequence) - 2):
            if sequence[i] == sequence[i+1] == sequence[i+2]:
                return False
        return True

    def generate_positive(self, max_length: int) -> List[float]:
        length = np.random.randint(self.min_len, max_length + 1)
        seq = []
        for _ in range(length):
            val = float(np.random.randint(0, 5))
            # Constructive avoidance
            if len(seq) >= 2 and seq[-1] == val and seq[-2] == val:
                val = float((val + 1) % 5)
            seq.append(val)
        return seq

class L5(Language):
    """Even count of A and B"""
    def check(self, sequence: List[float]) -> bool:
        sequence = list(sequence)  # Convert numpy array to list
        unique = list(set(sequence))
        if len(unique) != 2: return False
        return sequence.count(unique[0]) % 2 == 0 and sequence.count(unique[1]) % 2 == 0

    def generate_positive(self, max_length: int) -> List[float]:
        a, b = float(0), float(1)
        ca = 2 * np.random.randint(1, 5)
        cb = 2 * np.random.randint(1, 5)
        seq = [a]*ca + [b]*cb
        random.shuffle(seq)
        return seq

class L6(Language):
    """Difference of counts divisible by 3"""
    def check(self, seq: List[float]) -> bool:
        seq = list(seq)  # Convert numpy array to list
        unique = list(set(seq))
        if len(unique) != 2: return len(seq) % 3 == 0 if len(unique)==1 else False
        c1 = seq.count(unique[0])
        c2 = seq.count(unique[1])
        return abs(c1 - c2) % 3 == 0

    def generate_positive(self, max_length: int) -> List[float]:
        # c1 - c2 = 3k => c1 = c2 + 3k
        c2 = np.random.randint(1, 10)
        c1 = c2 + 3 * np.random.randint(0, 3)
        seq = [0.0]*c1 + [1.0]*c2
        random.shuffle(seq)
        return seq

class L7(Language):
    """Pattern a* b* a* b*"""
    def check(self, lst: List[float]) -> bool:
        if len(lst) == 0: return True
        lst = list(lst)  # Convert numpy array to list
        unique = list(set(lst))
        if len(unique) > 2: return False
        
        # Simplify check: compress runs. [a, a, b, b, a] -> [a, b, a]
        compressed = [lst[0]]
        for x in lst[1:]:
            if x != compressed[-1]: compressed.append(x)
            
        # Allowed transitions: a->b->a->b (max 4 segments)
        return len(compressed) <= 4

    def generate_positive(self, max_length: int) -> List[float]:
        a, b = float(0), float(1)
        segs = np.random.randint(1, 5) # 1 to 4 segments
        seq = []
        curr = a
        for _ in range(segs):
            count = np.random.randint(1, 5)
            seq.extend([curr]*count)
            curr = b if curr == a else a
        return seq

# ==========================================
# 4. 驗證與執行
# ==========================================

def verify_balance(name, pos, neg):
    print(f"\n--- {name} Check ---")
    if not pos or not neg:
        print("Error: Empty dataset")
        return

    # Length
    lp = np.mean([len(x) for x in pos])
    ln = np.mean([len(x) for x in neg])
    print(f"Avg Length: Pos={lp:.2f}, Neg={ln:.2f}")
    if abs(lp - ln) > 3: print("WARNING: Length Bias detected!")

    # Value Range
    vp = np.mean([np.mean(x) for x in pos])
    vn = np.mean([np.mean(x) for x in neg])
    print(f"Avg Value:  Pos={vp:.2f}, Neg={vn:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('language', nargs='?', default='S1')
    parser.add_argument('--num', type=int, default=500)
    args = parser.parse_args()

    # Map string to class
    globs = globals()
    if args.language in globs:
        cls = globs[args.language]
        lang = cls(num_pos=args.num, num_neg=args.num, max_len=20)
        pos, neg = lang.generate_samples(args.num, args.num, 20)
        
        verify_balance(args.language, pos, neg)
        
        print(f"Generated {len(pos)} Pos, {len(neg)} Neg.")
        print("Sample Pos:", pos[0])
        print("Sample Neg:", neg[0])
    else:
        print(f"Language {args.language} not found.")

if __name__ == "__main__":
    main()