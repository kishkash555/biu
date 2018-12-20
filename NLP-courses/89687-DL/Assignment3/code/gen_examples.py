#!/usr/bin/python
import sys
"""
automatically generate positive and negative examples for training an LSTM

Positive: [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+
Negative: [1-9]+a+[1-9]+c+[1-9]+b+[1-9]+d+[1-9]+

Usage:
    gen_examples.py <pos> <neg>
<pos>  number of positive examples to generate
<neg>  number of negative examples to generate
"""

min_repeat = 2
max_repeat = 5
n_letter_segments = 4
n_digit_segments = 5
positive = []
from numpy.random import randint
import numpy as np

def random_sequence(letter_sequence):
    lengths_letter_segments = randint(min_repeat, max_repeat, n_letter_segments)
    length_digit_segments = randint(min_repeat, max_repeat, n_digit_segments)
    random_digits = list(map(str,randint(1,10,sum(length_digit_segments))))

    next_dig_to_read = 0
    ret = []
    for i in range(4):
    # while digs < length_digit_segments or letts < len(letter_sequence):
        ret += random_digits[next_dig_to_read: next_dig_to_read+length_digit_segments[i]] + [letter_sequence[i]]*lengths_letter_segments[i]
        next_dig_to_read += length_digit_segments[i]
    
    return "".join(ret)
class part1_syntax:
    @classmethod
    def positive_sequence(cls):
        return random_sequence("abcd")
    @classmethod
    def negative_sequence(cls):
        return random_sequence("acbd")

class number_sequence():
    """

    """
    @classmethod
    def positive_sequence(cls):
        ret = []
        length_digit_segment = randint(min_repeat, max_repeat-1, 1)[0]
        random_digits = list(map(str,randint(1,10,length_digit_segment)))
        random_even_digit = str(randint(1,5)*2)
        ret = random_digits + [random_even_digit]
        if len(ret) >= 3:
            ret[2] = "a"#*len(ret[::4])
        else: 
            ret += ["a"]* (3-len(ret))
        return "".join(ret)

    @classmethod
    def negative_sequence(cls):
        ret = []
        length_digit_segment = randint(min_repeat, max_repeat-1, 1)[0]
        random_digits = list(map(str,randint(1,10,length_digit_segment)))
        random_even_digit = str(randint(1,6)*2-1)
        ret = random_digits + [random_even_digit]
        if len(ret) >= 8:
            ret[7] = "a"#*len(ret[::4])
        else: 
            ret += ["a"]* (8-len(ret))
        return "".join(ret)


class monotonic_sequence():
    @classmethod
    def positive_sequence(cls):
        ret = []
        length_digit_segment = randint(min_repeat, max_repeat, 1)[0]
        random_digits = randint(0,4,length_digit_segment)
        cm = 1
        reached9 = False
        for i in range(len(random_digits)):
            cm += random_digits[i]
            if cm >= 10:
                reached9 = True
                random_digits[i]=9
                cm = 1
            else:
                random_digits[i] = cm
        
        ret = list(map(str,random_digits))
        if not reached9:
            ret += ["9"]
        return "".join(ret)

    @classmethod
    def negative_sequence(cls):
        ret = []
        length_digit_segment = randint(min_repeat, max_repeat, 1)[0]
        random_digits = randint(1,10,length_digit_segment)
        
        neg_diff = np.diff(random_digits)<0
        neg_diff_val = random_digits[:-1][neg_diff]
        if np.all(neg_diff_val==9):
            if len(neg_diff_val)>0:
                random_digits[np.where(neg_diff_val)]=8
            elif random_digits[-1]==1:
                a = randint(0,length_digit_segment-1)
                b = randint(2,9)
                random_digits[a]=b
            else:
                b = randint(1,random_digits[-1])
                random_digits = np.concatenate([random_digits, [b]])

        ret = list(map(str,random_digits)) 
        return "".join(ret)

class multiples:
    @classmethod
    def positive_sequence(cls):
        ret = ""
        for i in range(n_digit_segments):
            num = randint(100,1000000)
            ret += str(num)
            if num % 7 <= 2:
                ret += "a"
            else:
                ret += "b"
        return ret
    @classmethod
    def negative_sequence(cls):
        ret = ""
        for i in range(n_digit_segments):
            num = randint(100,1000000)
            ret += str(num)
            if num % 7 <= 2:
                ret += "b"
            else:
                ret += "a"
        return ret

class even_sums:
    @classmethod
    def positive_sequence(cls):
        ret = ""
        for i in range(n_digit_segments):
            length_digit_segment = randint(min_repeat, max_repeat-1, 1)[0]
            random_digits = randint(1,10,length_digit_segment)
            if random_digits.sum() % 2 == 0:
                random_extra_digit = str(randint(1,5)*2)
            else:
                random_extra_digit = str(randint(1,5)*2-1)
            ret += "".join(list(map(str,random_digits))) + random_extra_digit + "a"
        return ret
    
    @classmethod
    def negative_sequence(cls):
        ret = ""
        for i in range(n_digit_segments):
            length_digit_segment = randint(min_repeat, max_repeat-1, 1)[0]
            random_digits = randint(1,10,length_digit_segment)
            if random_digits.sum() % 2 == 0:
                random_extra_digit = str(randint(1,5)*2-1)
            else:
                random_extra_digit = str(randint(1,5)*2)
            ret += "".join(list(map(str,random_digits))) + random_extra_digit + "a"
        return ret

class match_start:
    @classmethod
    def positive_sequence(cls):
        ret = ""
        length_digit_segment = randint(4,20)
        random_digits = randint(1,10, length_digit_segment)
        letter_segment = "a" *random_digits[0] + \
            "b" * random_digits[1] + \
            "c" * random_digits[2] + \
            "d" * random_digits[3]
        ret += "".join(list(map(str,random_digits))) + letter_segment
        return ret
    
    @classmethod
    def negative_sequence(cls):
        ret = ""
        length_digit_segment = randint(4,20)
        random_digits = randint(1,10, length_digit_segment)
        if random_digits[0] == random_digits[3] and random_digits[1]==random_digits[2]:
            random_digits[3] = random_digits[3] % 8 + 1 # alter
        letter_segment = "a" *random_digits[3] + \
            "b" * random_digits[2] + \
            "c" * random_digits[1] + \
            "d" * random_digits[0]
        ret += "".join(list(map(str,random_digits))) + letter_segment
        return ret
    
class simple_counter:
    @classmethod
    def positive_sequence(cls):
        ret = []
        number_of_segments = randint(4,20)
        random_digits = randint(1,10, number_of_segments)
        for d in random_digits:
            letters = randint(0,3,d)
            letters_list = map(lambda l: "abcd"[l], list(letters))
            ret += letters_list + [str(d)]
        ret = "".join(ret)
        return ret

    @classmethod
    def negative_sequence(cls):
        ret = []
        number_of_segments = randint(4,20)
        random_digits = randint(1,10, number_of_segments)
        for d in random_digits:
            letters = randint(0,3,11-d)
            letters_list = map(lambda l: "abcd"[l], list(letters))
            ret += letters_list + [str(d)]
        ret = "".join(ret)
        return ret
    

if __name__ == "__main__":
    argv = sys.argv
    try:
        n_positive = int(argv[1])
    except:
        n_positive = 10
    try:
        n_negative = int(argv[2])
    except:
        n_negative = 10
    try: 
        max_repeat = int(argv[3])
    except: 
        max_repeat = 5
    
    gen_class = simple_counter
    for j in range(n_positive):
        print("1\t" + gen_class.positive_sequence())

    for j in range(n_negative):
        print("0\t" + gen_class.negative_sequence())

