import utils
from collections import Counter
import numpy as np


def create_symbol_dict(symbols):
    return {sy: i for i, sy in enumerate(symbols)}

def generate_vector(s, symbol_counts, location_dict):
    x = np.zeros(s, np.double)
    for symb, cnt in symbol_counts:
        if symb in location_dict and location_dict[symb] < s: # discard symbol if not known
            x[location_dict[symb]]=cnt
    return x


