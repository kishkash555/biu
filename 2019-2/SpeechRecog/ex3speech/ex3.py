import numpy as np
import argparse
from collections import namedtuple
from ex3_test import prob_calc_bf
from os import path
#debug = True

debug_args_nt = namedtuple('debug_args',['file','labeling','alphabet'])
debug_args = debug_args_nt('', 'ace', 'abcdef')

probs = np.array([
       [0.4 , 0.06, 0.01, 0.  , 0.  , 0.1 , 0.  , 0.02, 0.03, 0.03],
       [0.16, 0.2 , 0.02, 0.05, 0.  , 0.1 , 0.05, 0.02, 0.01, 0.1 ],
       [0.03, 0.01, 0.09, 0.24, 0.55, 0.27, 0.25, 0.4 , 0.17, 0.06],
       [0.17, 0.  , 0.38, 0.16, 0.05, 0.14, 0.15, 0.05, 0.08, 0.66],
       [0.1 , 0.18, 0.3 , 0.  , 0.01, 0.1 , 0.15, 0.5 , 0.  , 0.11],
       [0.1 , 0.2 , 0.  , 0.03, 0.09, 0.01, 0.  , 0.  , 0.46, 0.  ],
       [0.04, 0.35, 0.2 , 0.52, 0.3 , 0.28, 0.4 , 0.01, 0.25, 0.04]])

probs1 = np.array([
    [1,0,0,0,0,0,0],
    [1,0,0,0,0,0,0],
    [.5,0,0,0,0,0,.5],
    [0,0,.5,0,0,0,.5],
    [0,0,0,0,0,0,1],
    [0,0,0,0,.5,0,.5],
]).T



def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("file", type=str, help='path to numpy binary with input probabilities')
    parser.add_argument("labeling", type=str, help='the labeling you wish to calculate probability for')
    parser.add_argument("alphabet", type=str, help='the output tokens corresponding to the columns of the matrix')
    return parser.parse_args()

def main():
    args = parseargs()
    debug = args.debug
    prob_mat = None
    if path.isfile(args.file):
        prob_mat = np.load(args.file).T
    if debug:
        prob_mat = prob_mat if prob_mat is not None else probs
        args.labeling = args.labeling if args.labeling != '-' else debug_args.labeling
        args.alphabet = args.alphabet if args.alphabet != '-' else debug_args.alphabet
        bruteforce = prob_calc_bf(prob_mat, args.labeling, args.alphabet)
        print('brute force probability: {}'.format(bruteforce) )

    num_tokens = len(args.alphabet)+1
    if prob_mat.shape[0] != num_tokens:
        raise ValueError("input matrix has {} rows, alphabet has {} tokens".format(prob_mat.shape[0], num_tokens-1 ))
    prbty = ctc_probl_calc(prob_mat, args.labeling, args.alphabet)
    
    print('{:.2}'.format(round(prbty,2)))




def shift(a):
    return np.concatenate([np.zeros(1),a[:-1]])

def ctc_probl_calc(y, word, alphabet):
    label_length = len(word)
    alpha = np.zeros(label_length * 2 + 1)
    alphabet_dict = { t: i for i, t in enumerate(alphabet)}
    blank_ordinal = len(alphabet)
    z = [blank_ordinal] + sum([[alphabet_dict[token], blank_ordinal] for token in word],[])

    allow_skip = np.zeros(label_length*2 +1, dtype=bool)
    allow_skip[1::2] = [True] + [l1 != l2 for l1, l2 in zip(word[1:], word[:-1])]
    
    # initial conditions
    alpha[0:2] = y[z[0:2],0]

    # recursion
    for t in range(1,y.shape[1]):
        alpha_1 = shift(alpha)
        alpha_2 = shift(alpha_1)
        alpha = (alpha + alpha_1 + allow_skip * alpha_2) * y[z,t]

    return alpha[-2:].sum()

if __name__ == "__main__":
    main()