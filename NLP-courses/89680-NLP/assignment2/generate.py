"""Generate sentences

Usage: 
    generate.py <file> [[--tree -n<num>]|--parse=<sentence>]

Options:
    <file>  name of grammar file
    --tree  output the rule tree with the sentence
    -n <num>  number of sentences to generate
    --parse <sentence>  check if the sentence can be parsed using the current grammar
"""

from docopt import docopt
from collections import defaultdict
import random

class PCFG(object):
    def __init__(self):
        self._rules = defaultdict(list)
        self._sums = defaultdict(float)

    def add_rule(self, lhs, rhs, weight):
        assert(isinstance(lhs, str))
        assert(isinstance(rhs, list))
        self._rules[lhs].append((rhs, weight))
        self._sums[lhs] += weight

    @classmethod
    def from_file(cls, filename):
        grammar = PCFG()
        with open(filename) as fh:
            for line in fh:
                line = line.split("#")[0].strip()
                if not line: continue
                w,l,r = line.split(None, 2)
                r = r.split()
                w = float(w)
                grammar.add_rule(l,r,w)
        return grammar

    def is_terminal(self, symbol): return symbol not in self._rules

    def gen(self, symbol):
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion(symbol)
            return " ".join(self.gen(s) for s in expansion)

    def random_sent(self):
        return self.gen("ROOT")

    def random_expansion(self, symbol):
        """
        Generates a random RHS for symbol, in proportion to the weights.
        """
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0: return r
        return r

    def random_sent_t(self):
        ret = self.gen_t("ROOT")
        return pretty_tree(ret,0)

    def gen_t(self, symbol):
        if self.is_terminal(symbol): return symbol
        else:
            expansion = self.random_expansion_t(symbol)
            return (expansion[0], [self.gen_t(s) for s in expansion[1]])
    
    def random_expansion_t(self, symbol):
        p = random.random() * self._sums[symbol]
        for r,w in self._rules[symbol]:
            p = p - w
            if p < 0:
                 return (symbol,r)
        return (symbol,r)

    def parse(self, words):
        if type(words)==str:
            words = words.split(" ")
        if len(words)==1 and words[0]=="ROOT":
            return True
        for k, v in self._rules.iteritems():
            lhs = k.split()
            for rhs,_ in v:
                start = -1
                start, ln = self.find_match(words, rhs, start)
                while start >=0 :
                    next_sent = words[0:start] + lhs + words[start+ln:]
                    #print(next_sent)
                    if self.parse(next_sent):
                        return True
                    start, ln = self.find_match(words, rhs, start+1)
                
        return False

    def find_match(self, words, rhs, start_pos):
        if start_pos == -1: start_pos = 0
        for i in range(start_pos,len(words)):
            m = 0 
            while m < len(rhs) and words[i+m] == rhs[m]: m+=1
            if m == len(rhs):
                return i , m
        return -1, 0

def simplify(expr):
    if type(expr)==str: return expr
    simplified = tuple(simplify(x) for x in expr[1][0])
    return (expr[0], simplified)

def pretty_tree(expr, indent):
    """
    print the tree represented by expr elegantly
    """
    if type(expr)==str:
        return expr
    indent += len(expr[0])+2
    ret="({} {})".format(expr[0], ("\n"+" "*indent).join([pretty_tree(e, indent) for e in expr[1]]))
    return ret

if __name__ == '__main__':
    arguments = docopt(__doc__)
    pcfg = PCFG.from_file(arguments['<file>'])
    print arguments
    if '--parse' in arguments and arguments['--parse']:
        sent = arguments['--parse']
        print pcfg.parse(sent)
        
    else:
        try:
            reps = int(arguments['-n'])
        except:
            reps = 1
        
        gen_func = pcfg.random_sent_t if arguments['--tree'] else pcfg.random_sent
        for r in range(reps):
            print
            print gen_func()


   