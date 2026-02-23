
import random
# from typing import List, Tuple, Callable, Dict
import numpy as np

# def _repeat_block(block, repeats):
#     return block * repeats

# def _rand_block(block_len, distinct):
#     # distinct=1: all a or all b; distinct=2: a,b
#     if distinct==1:
#         c = np.random.choice(['a','b'])
#         return [c]*block_len
#     else:
#         return np.random.permutation(['a','b']* ((block_len+1)//2))[:block_len].tolist()

# def _rand_seq(l):
#     return np.random.choice(['a','b','c','d'], l).tolist()

# def _make_pattern(block, repeats):
#     return block*repeats

# def _make_neg_pattern(block, repeats):
#     seq = block*repeats
#     if seq:
#         seq[0] = 'a' if seq[0]=='b' else 'b'
#     return seq

# def _even(n):
#     return n%2==0

# def _odd(n):
#     return n%2==1

# def _mod(n, m):
#     return n%m==0

# def _make_block(n, c):
#     return [c]*n

# def _make_abab(n):
#     return ['a','b']*(n//2)

# def _to_ab(seq):
#     if isinstance(seq, np.ndarray):
#         seq = seq.tolist()
#     if seq is None or len(seq) == 0:
#         return []
#     if isinstance(seq[0], int):
#         return ['a' if x == 0 else 'b' for x in seq]
#     return [str(x) for x in seq]

class L1:
    # Accepts only constant sequences (all symbols the same). Works for any alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     return all(x == s[0] for x in s)
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
    # Accepts only strings of the form (abcd)* (i.e., a,b,c,d,a,b,c,d,...)
    # def check(self, lst):
    #     if len(lst) % 4 != 0:
    #         return False
    #     for i in range(0, len(lst), 4):
    #         if lst[i:i+4] != ['a','b','c','d']:
    #             return False
    #     return True
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
    # Accepts only strings where every block of 2 symbols is the same (e.g., aa, bb, cc, dd, ...).
    # def check(self, lst):
    #     s = lst
    #     if len(s)%2 != 0:
    #         return False
    #     for i in range(0, len(s), 2):
    #         if not all(x==s[i] for x in s[i:i+2]):
    #             return False
    #     return True
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
    # Accepts only strings of the form (abcd abcd ...)*.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%4!=0:
    #         return False
    #     for i in range(0,len(s),4):
    #         if not (s[i]=='a' and s[i+1]=='b' and s[i+2]=='a' and s[i+3]=='b'):
    #             return False
    #     return True
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
    # Accepts strings where the count of 'a' is odd and 'b' is even, or vice versa.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     ca = s.count('a')
    #     cb = s.count('b')
    #     return (ca%2==1 and cb%2==0) or (cb%2==1 and ca%2==0)
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
    # Accepts only strings of the form (abcd abcd abcd)*.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%6!=0:
    #         return False
    #     for i in range(0,len(s),6):
    #         if not (s[i]=='a' and s[i+1]=='b' and s[i+2]=='a' and s[i+3]=='b' and s[i+4]=='a' and s[i+5]=='b'):
    #             return False
    #     return True
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            r = np.random.randint(1, max_length//6+1)
            pos.append(['a','b','c','d','a','b','c','d']*r)
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
    
class L4:
    # Accepts strings with no three consecutive identical symbols.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     for i in range(len(s)-2):
    #         if s[i] == s[i+1] == s[i+2]:
    #             return False
    #     return True
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            l = np.random.randint(3, max_length+1)
            seq = []
            for _ in range(l):
                if len(seq)>=2 and seq[-1]==seq[-2]:
                    c = np.random.choice([x for x in alphabet if x != seq[-1]])
                else:
                    c = np.random.choice(alphabet)
                seq.append(c)
            pos.append(seq)
        for _ in range(num_neg):
            l = np.random.randint(3, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            if any(seq[i]==seq[i+1]==seq[i+2] for i in range(l-2)):
                neg.append(seq)
            else:
                for j in range(min(3, l)):
                    seq[j] = np.random.choice([x for x in alphabet if x != seq[j]])
                neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg
    
class L4A:
    # Accepts only strings where every block of 4 symbols is the same.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%4!=0:
    #         return False
    #     for i in range(0,len(s),4):
    #         if not all(x==s[i] for x in s[i:i+4]):
    #             return False
    #     return True
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
    # Accepts only strings of the form (abcd abcd abcd abcd)*.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%8!=0:
    #         return False
    #     for i in range(0,len(s),8):
    #         if not (s[i]=='a' and s[i+1]=='b' and s[i+2]=='a' and s[i+3]=='b' and s[i+4]=='a' and s[i+5]=='b' and s[i+6]=='a' and s[i+7]=='b'):
    #             return False
    #     return True
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
    # Accepts strings where the count of 'a' and 'b' are both even. Only for a,b alphabet. NOT for 4-symbol alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     ca = s.count('a')
    #     cb = s.count('b')
    #     return _even(ca) and _even(cb)
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            ca = 2*np.random.randint(1,5)
            cb = 2*np.random.randint(1,5)
            rest = np.random.randint(0, 3)
            seq = ['a']*ca + ['b']*cb + [np.random.choice(['c','d']) for _ in range(rest)]
            np.random.shuffle(seq)
            pos.append(seq)
        for _ in range(num_neg):
            ca = 2*np.random.randint(1,5)+1
            cb = 2*np.random.randint(1,5)
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

class L5A:
    # Accepts only strings where every block of 5 symbols is the same. Works for any alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%5!=0:
    #         return False
    #     for i in range(0,len(s),5):
    #         if not all(x==s[i] for x in s[i:i+5]):
    #             return False
    #     return True
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
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%10!=0:
    #         return False
    #     for i in range(0,len(s),10):
    #         if not (s[i]=='a' and s[i+1]=='b' and s[i+2]=='a' and s[i+3]=='b' and s[i+4]=='a' and s[i+5]=='b' and s[i+6]=='a' and s[i+7]=='b' and s[i+8]=='a' and s[i+9]=='b'):
    #             return False
    #     return True
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
    # Accepts strings where |#a - #b| mod 3 == 0. Only for a,b alphabet. NOT for 4-symbol alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     ca = s.count('a')
    #     cb = s.count('b')
    #     return _mod(abs(ca-cb),3)
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            ca = 3*np.random.randint(1,5)
            cb = ca + 3*np.random.randint(0,3)
            rest = np.random.randint(0, 3)
            seq = ['a']*ca + ['b']*cb + [np.random.choice(['c','d']) for _ in range(rest)]
            np.random.shuffle(seq)
            pos.append(seq)
        for _ in range(num_neg):
            ca = 3*np.random.randint(1,5)+1
            cb = ca + 3*np.random.randint(0,3)
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

class L6A:
    # Accepts only strings where every block of 6 symbols is the same. Works for any alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%6!=0:
    #         return False
    #     for i in range(0,len(s),6):
    #         if not all(x==s[i] for x in s[i:i+6]):
    #             return False
    #     return True
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
    # Accepts strings of the form a^n b^m a^k b^l. Only for a,b alphabet. NOT for 4-symbol alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     i = 0
    #     n = len(s)
    #     while i<n and s[i]=='a': i+=1
    #     while i<n and s[i]=='b': i+=1
    #     while i<n and s[i]=='a': i+=1
    #     while i<n and s[i]=='b': i+=1
    #     return i==n
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        for _ in range(num_pos):
            l1 = np.random.randint(1, max_length//4+1)
            l2 = np.random.randint(1, max_length//4+1)
            l3 = np.random.randint(1, max_length//4+1)
            l4 = np.random.randint(1, max_length//4+1)
            seq = ['a']*l1 + ['b']*l2 + ['a']*l3 + ['b']*l4
            pos.append(seq)
        for _ in range(num_neg):
            l = np.random.randint(4, max_length+1)
            seq = np.random.choice(alphabet, l).tolist()
            neg.append(seq)
        def to_str_list(seq):
            if isinstance(seq, tuple):
                return [str(x) for x in seq]
            return [str(x) for x in list(seq)]
        pos = [to_str_list(seq) for seq in pos]
        neg = [to_str_list(seq) for seq in neg]
        return pos, neg

class L7A:
    # Accepts only strings where every block of 7 symbols is the same. Works for any alphabet.
    # def check(self, lst):
    #     s = _to_abcd(lst)
    #     if len(s)%7!=0:
    #         return False
    #     for i in range(0,len(s),7):
    #         if not all(x==s[i] for x in s[i:i+7]):
    #             return False
    #     return True
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
    # Accepts only strings where every block of 8 symbols is the same. Works for any alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%8!=0:
    #         return False
    #     for i in range(0,len(s),8):
    #         if not all(x==s[i] for x in s[i:i+8]):
    #             return False
    #     return True
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
    # Accepts only strings where every block of 9 symbols is the same. Works for any alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%9!=0:
    #         return False
    #     for i in range(0,len(s),9):
    #         if not all(x==s[i] for x in s[i:i+9]):
    #             return False
    #     return True
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
    # Accepts only strings where every block of 10 symbols is the same. Works for any alphabet.
    # def check(self, lst):
    #     s = _to_ab(lst)
    #     if len(s)%10!=0:
    #         return False
    #     for i in range(0,len(s),10):
    #         if not all(x==s[i] for x in s[i:i+10]):
    #             return False
    #     return True
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

# class EvenPairs:
#     def check(self, seq):
#         # seq: list of 'a'/'b'/'c'
#         s = ''.join(seq)
#         c_ab = sum(1 for i in range(len(s)-1) if s[i:i+2] == 'ab')
#         c_ba = sum(1 for i in range(len(s)-1) if s[i:i+2] == 'ba')
#         # ignore pairs with 'c'
#         return (c_ab + c_ba) % 2 == 0
#     def generate_samples(self, num_pos, num_neg, max_length):
#         pos, neg = [], []
#         alphabet = ['a', 'b', 'c', 'd']
#         while len(pos) < num_pos or len(neg) < num_neg:
#             l = random.randint(2, max_length)
#             s = [random.choice(alphabet) for _ in range(l)]
#             if self.check(s):
#                 if len(pos) < num_pos:
#                     pos.append(s)
#             else:
#                 if len(neg) < num_neg:
#                     neg.append(s)
#         return pos, neg
