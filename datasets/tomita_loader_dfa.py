
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
    
class L4:
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []
        alphabet = ['a','b','c','d']
        # ----- Positive -----
        while len(pos) < num_pos:
            length = np.random.randint(3, max_length + 1)
            seq = []
            for _ in range(length):
                if len(seq) >= 2 and seq[-1] == seq[-2]:
                    choices = [x for x in alphabet if x != seq[-1]]
                    seq.append(np.random.choice(choices))
                else:
                    seq.append(np.random.choice(alphabet))
            pos.append(seq)

        # ----- Negative -----
        while len(neg) < num_neg:
            length = np.random.randint(3, max_length + 1)
            seq = np.random.choice(alphabet, length).tolist()

            # 強制插入 triple
            pos_insert = np.random.randint(0, length - 2)
            symbol = np.random.choice(alphabet)
            seq[pos_insert:pos_insert+3] = [symbol]*3

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
    # Accepts strings where a/b sequence is a*b*a*b* (ignoring c/d noise).
    # Positive examples: pure a/b in a*b*a*b* structure (no c/d)
    # Negative examples: may contain c/d noise
    def generate_samples(self, num_pos, num_neg, max_length):
        pos, neg = [], []

        for _ in range(num_pos):
            l1 = np.random.randint(1, max_length//4)
            l2 = np.random.randint(1, max_length//4)
            l3 = np.random.randint(1, max_length//4)
            l4 = np.random.randint(1, max_length//4)

            base = ['a']*l1 + ['b']*l2 + ['a']*l3 + ['b']*l4
            
            # 插入 noise
            seq = []
            for s in base:
                seq.append(s)
                if np.random.rand() < 0.2:
                    seq.append(np.random.choice(['c','d']))
            
            pos.append(seq)

        for _ in range(num_neg):
            # 專門破壞 phase
            l1 = np.random.randint(1, max_length//4)
            l2 = np.random.randint(1, max_length//4)
            l3 = np.random.randint(1, max_length//4)

            seq = ['a']*l1 + ['b']*l2 + ['a']*l3 + ['b']*l2 + ['a']  # extra a

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
