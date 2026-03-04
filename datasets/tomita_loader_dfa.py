
from fontTools.unicodedata import normalize
import numpy as np

class L1:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            l = np.random.randint(1, max_length+1)
            c = np.random.choice(alphabet)
            pos.append([c]*l)
        for _ in range(num_neg):
            l = np.random.randint(2, max_length+1)
            c = np.random.choice(alphabet)
            seq = [c]*l
            # introduce at least one different character
            idx = np.random.randint(0, l)
            other = np.random.choice([x for x in alphabet if x != c])
            seq[idx] = other
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L2:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//4+1)
            pos.append(['a','b','c','d']*r)
        for _ in range(num_neg):
            l = np.random.randint(1, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            # If by chance it is a valid (abcd)*, break it
            if l % 4 == 0 and all(seq[i:i+4]==['a','b','c','d'] for i in range(0,l,4)):
                seq[0] = 'b'
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L2A:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//2+1)
            c = np.random.choice(alphabet)
            pos.append([c]*2*r)
        for _ in range(num_neg):
            l = np.random.randint(2, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%2==0 and all(all(x==seq[i] for x in seq[i:i+2]) for i in range(0,len(seq),2)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L2AB:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//4+1)
            pos.append(['a','b','a','b']*r)
        for _ in range(num_neg):
            l = np.random.randint(4, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%4==0 and all((seq[i]=='a' and seq[i+1]=='b' and seq[i+2]=='a' and seq[i+3]=='b') for i in range(0,len(seq),4)):
                seq[0] = 'c'
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg
    
class L3:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        # alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            ca = 2*np.random.randint(1,5)+1
            cb = 2*np.random.randint(0,5)
            rest = np.random.randint(0, 3)
            seq = ['a']*ca + ['b']*cb + [np.random.choice(['c','d']) for _ in range(rest)]
            np.random.shuffle(seq)
            pos.append(seq)
        for _ in range(num_neg):
            ca = 2*np.random.randint(0,5)
            cb = 2*np.random.randint(0,5)
            rest = np.random.randint(0, 3)
            seq = ['a']*ca + ['b']*cb + [np.random.choice(['c','d']) for _ in range(rest)]
            np.random.shuffle(seq)
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L3A:
    # Accepts only strings where every block of 3 symbols is the same (e.g., aaa, bbb, ccc, ddd, ...).
    def check(self, lst):
        s = lst
        if len(s)%3 != 0:
            return False
        for i in range(0, len(s), 3):
            if not all(x==s[i] for x in s[i:i+3]):
                return False
        return True
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//3+1)
            c = np.random.choice(alphabet)
            pos.append([c]*3*r)
        for _ in range(num_neg):
            l = np.random.randint(3, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%3==0 and all(all(x==seq[i] for x in seq[i:i+3]) for i in range(0,len(seq),3)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L3AB:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//6+1)
            pos.append(['a','b','c','d','a','b','c','d','a','b','c','d']*r)
        for _ in range(num_neg):
            l = np.random.randint(6, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%6==0 and all((seq[i]=='a' and seq[i+1]=='b' and seq[i+2]=='c' and seq[i+3]=='d') for i in range(0,len(seq),4)):
                seq[0] = 'c'
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg
    
import random

class L4:
    # Language: No three consecutive identical symbols
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a', 'b', 'c', 'd']

        # ---------- Generate Positive ----------
        def generate_positive():
            length = random.randint(3, max_length)
            seq = []
            for _ in range(length):
                if len(seq) >= 2 and seq[-1] == seq[-2]:
                    choices = [x for x in alphabet if x != seq[-1]]
                    seq.append(random.choice(choices))
                else:
                    seq.append(random.choice(alphabet))
            return seq

        # Positive samples
        while len(pos) < num_pos:
            pos.append(generate_positive())

        # ---------- Generate Negative ----------
        while len(neg) < num_neg:
            # Step 1: start from a valid positive
            seq = generate_positive()

            # Step 2: force a violation
            if len(seq) >= 3:
                insert_pos = random.randint(0, len(seq) - 3)
                symbol = random.choice(alphabet)
                seq[insert_pos:insert_pos+3] = [symbol, symbol, symbol]

                # Ensure violation really exists
                # (in rare case overwrite didn't change anything)
                if any(seq[i] == seq[i+1] == seq[i+2] 
                       for i in range(len(seq)-2)):
                    neg.append(seq)

        return pos, neg
    
class L4A:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a', 'b', 'c', 'd']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//4+1)
            c = np.random.choice(alphabet)
            pos.append([c]*4*r)
        for _ in range(num_neg):
            l = np.random.randint(4, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%4==0 and all(all(x==seq[i] for x in seq[i:i+4]) for i in range(0,len(seq),4)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg
    
class L4AB:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//8+1)
            pos.append(['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd', 'a' ,'b' ,'c' ,'d']*r)
        for _ in range(num_neg):
            l = np.random.randint(8, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%8==0 and all((seq[i]=='a' and seq[i+1]=='b' and seq[i+2]=='c' and seq[i+3]=='d') for i in range(0,len(seq),4)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L5:
    # Accepts strings where the count of 'a' and 'b' are both even.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        for _ in range(num_pos):
            # Both ca and cb are even, and ca + cb <= max_length
            ca = 2 * np.random.randint(1, max(2, max_length // 2))
            max_cb = (max_length - ca) // 2
            cb = 2 * np.random.randint(1, max(2, max_cb + 1))
            if ca + cb <= max_length:
                seq = ['a']*ca + ['b']*cb
                np.random.shuffle(seq)
                pos.append(seq)
        for _ in range(num_neg):
            # One of ca or cb is odd, and ca + cb <= max_length
            if np.random.rand() < 0.5:
                # ca is odd, cb is even
                ca = 2*np.random.randint(0, max_length // 2) + 1
                max_cb = (max_length - ca) // 2
                cb = 2 * np.random.randint(1, max(2, max_cb + 1)) if max_cb >= 1 else 0
            else:
                # ca is even, cb is odd
                ca = 2 * np.random.randint(1, max(2, max_length // 2))
                max_cb = (max_length - ca - 1) // 2
                cb = 2 * np.random.randint(0, max(1, max_cb + 1)) + 1
            if ca > 0 and cb > 0 and ca + cb <= max_length:
                seq = ['a']*ca + ['b']*cb
                np.random.shuffle(seq)
                neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L5A:
    # Accepts only strings where every block of 5 symbols is the same. Works for any alphabet.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//5+1)
            c = np.random.choice(alphabet)
            pos.append([c]*5*r)
        for _ in range(num_neg):
            l = np.random.randint(5, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%5==0 and all(all(x==seq[i] for x in seq[i:i+5]) for i in range(0,len(seq),5)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg
    
class L5AB:
    # Accepts only strings of the form (ab ab ab ab ab)*. Only works for a,b alphabet. NOT for 4-symbol alphabet.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            high = max_length // 10 + 1
            if high <= 1:
                r = 1
            else:
                r = np.random.randint(1, high)
            pos.append(['a','b','a','b','a','b','a','b','a','b']*r)
        for _ in range(num_neg):
            if max_length < 10:
                l = 10
            else:
                l = np.random.randint(10, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%10==0 and all((seq[i]=='a' and seq[i+1]=='b' and seq[i+2]=='a' and seq[i+3]=='b' and seq[i+4]=='a' and seq[i+5]=='b' and seq[i+6]=='a' and seq[i+7]=='b' and seq[i+8]=='a' and seq[i+9]=='b') for i in range(0,len(seq),10)):
                seq[0] = 'c'
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L6:
    # Accepts strings where |#a - #b| mod 3 == 0, i.e., #a ≡ #b (mod 3).
    # Sequences may contain 'c' and 'd' as noise, but rule only checks 'a' and 'b'.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet_noise = ['c', 'd']

        def random_seq_with_counts(ca, cb, length):
            seq = ['a'] * ca + ['b'] * cb
            remaining = length - len(seq)
            if remaining > 0:
                seq += np.random.choice(alphabet_noise, remaining).tolist()
            np.random.shuffle(seq)
            return seq

        # ----- Positive -----
        while len(pos) < num_pos:
            length = np.random.randint(1, max_length + 1)

            ca = np.random.randint(0, length + 1)
            cb = np.random.randint(0, length - ca + 1)

            if (ca - cb) % 3 == 0:
                pos.append(random_seq_with_counts(ca, cb, length))

        # ----- Negative -----
        while len(neg) < num_neg:
            length = np.random.randint(1, max_length + 1)

            ca = np.random.randint(0, length + 1)
            cb = np.random.randint(0, length - ca + 1)

            if (ca - cb) % 3 != 0:
                neg.append(random_seq_with_counts(ca, cb, length))

        return pos, neg

class L6A:
    # Accepts only strings where every block of 6 symbols is the same. Works for any alphabet.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//6+1)
            c = np.random.choice(alphabet)
            pos.append([c]*6*r)
        for _ in range(num_neg):
            l = np.random.randint(6, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%6==0 and all(all(x==seq[i] for x in seq[i:i+6]) for i in range(0,len(seq),6)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L7:
    # Accepts a*b*a*b* ignoring c/d noise
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        noise_alphabet = ['c', 'd']

        def insert_noise(seq, max_length):
            remaining = max_length - len(seq)
            noise_count = np.random.randint(0, remaining + 1)
            for _ in range(noise_count):
                idx = np.random.randint(0, len(seq)+1)
                seq.insert(idx, str(np.random.choice(noise_alphabet)))
            return seq

        # -------- Positive --------
        for _ in range(num_pos):
            l1 = np.random.randint(1, max_length//4 + 1)
            l2 = np.random.randint(1, max_length//4 + 1)
            l3 = np.random.randint(1, max_length//4 + 1)
            l4 = np.random.randint(1, max_length//4 + 1)

            seq = ['a']*l1 + ['b']*l2 + ['a']*l3 + ['b']*l4
            if len(seq) <= max_length:
                seq = insert_noise(seq, max_length)
                pos.append(seq)

        # -------- Negative --------
        for _ in range(num_neg):
            # break phase: add a after final b*
            l1 = np.random.randint(1, max_length//3 + 1)
            l2 = np.random.randint(1, max_length//3 + 1)
            l3 = np.random.randint(1, max_length//3 + 1)

            seq = ['a']*l1 + ['b']*l2 + ['a']*l3 + ['b']*l2 + ['a']
            if len(seq) <= max_length:
                seq = insert_noise(seq, max_length)
                neg.append(seq)

        return pos, neg

class L7A:
    # Accepts only strings where every block of 7 symbols is the same.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//7+1)
            c = np.random.choice(alphabet)
            pos.append([c]*7*r)
        for _ in range(num_neg):
            l = np.random.randint(7, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%7==0 and all(all(x==seq[i] for x in seq[i:i+7]) for i in range(0,len(seq),7)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg
    
class L8A:
    # Accepts only strings where every block of 8 symbols is the same.
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//8+1)
            c = np.random.choice(alphabet)
            pos.append([c]*8*r)
        for _ in range(num_neg):
            l = np.random.randint(8, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%8==0 and all(all(x==seq[i] for x in seq[i:i+8]) for i in range(0,len(seq),8)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L9A:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//9+1)
            c = np.random.choice(alphabet)
            pos.append([c]*9*r)
        for _ in range(num_neg):
            l = np.random.randint(9, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%9==0 and all(all(x==seq[i] for x in seq[i:i+9]) for i in range(0,len(seq),9)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L10A:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            high = max_length // 10 + 1
            if high <= 1:
                r = 1
            else:
                r = np.random.randint(1, high)
            c = np.random.choice(alphabet)
            pos.append([c]*10*r)
        for _ in range(num_neg):
            l = np.random.randint(10, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if len(seq)%10==0 and all(all(x==seq[i] for x in seq[i:i+10]) for i in range(0,len(seq),10)):
                seq[0] = np.random.choice([x for x in alphabet if x != seq[0]])
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg