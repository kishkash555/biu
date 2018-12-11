#!/usr/bin/python
import sys
"""
automatically generate positive and negative examples for training an LSTM

Positive: [1-9]+a+[1-9]+b+[1-9]+c+[1-9]+d+[1-9]+
"""

min_repeat = 1
max_repeat = 5
n_letter_segments = 4
n_digit_segments = 5
positive = []
from numpy.random import randint

def random_sequence(letter_sequence):
    lengths_letter_segments = randint(min_repeat, max_repeat, n_letter_segments)
    length_digit_segments = randint(min_repeat, max_repeat, n_digit_segments)
    random_digits = list(map(str,randint(1,9,sum(length_digit_segments))))

    next_dig_to_read = 0
    ret = []
    for i in range(4):
    # while digs < length_digit_segments or letts < len(letter_sequence):
        ret += random_digits[next_dig_to_read: next_dig_to_read+length_digit_segments[i]] + [letter_sequence[i]]*lengths_letter_segments[i]
        next_dig_to_read += length_digit_segments[i]
    
    return "".join(ret)

def positive_sequence():
    return random_sequence("abcd")

def negative_sequence():
    return random_sequence("acbd")

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
    
    for j in range(n_positive):
        print("1\t"+positive_sequence())

    for j in range(n_negative):
        print("0\t"+negative_sequence())

