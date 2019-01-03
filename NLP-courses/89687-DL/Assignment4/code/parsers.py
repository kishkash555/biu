import numpy as np
import csv
from collections import namedtuple
import re

SNLI_UNDECIDED_LABEL = "-"

class glove_embeddings:
    def __init__(self, glove_lines):
        self.embeddings = self.parse_glove_lines(glove_lines)
        self.vec_dim = next(self.embeddings.itervalues()).shape[0]
        print self.embeddings.keys()[100:110]
        self.unk_vec = self.embeddings['<unk>']
        self.word_to_ind = {word: i for i, word in enumerate(self.embeddings.keys())}
        self.unk_ord = self.word_to_ind['<unk>']

    @classmethod
    def parse_glove_lines(cls, glove_lines):
        ret = {}
        for line in glove_lines:
            word, data = line.strip().split(' ',1)
            ret[word]= np.fromstring(data, sep=' ') 
        return ret
    
    def get(self, word):
        return self.embeddings.get(word, self.unk_vec)
    
    def get_ord(self,word):
        return self.word_to_ind.get(word, self.unk_ord)
    
    def as_numpy_array(self):
        return np.array(self.embeddings.values())

    def as_dynet_lookup(self,pc):
        return pc.add_lookup_parameters((len(self.embeddings),self.vec_dim),  init = self.as_numpy_array())

def load_snli(fname, max_lines=0, is_separate_marks=True, remove_undecided = True, labels = None):
    data = []
    labels = labels or {}
    label_cnt = len(labels)
    cnt = 1
    with open(fname,'rt') as a:
        lines = csv.reader(a, delimiter='\t')
        row_nt = namedtuple('row',next(lines)) # take header
        for line in lines:
            row = row_nt(*line)
            if is_separate_marks:
                row = separate_marks(row)
            if row.gold_label == SNLI_UNDECIDED_LABEL and remove_undecided:
                continue
            if row.gold_label not in labels:
                labels[row.gold_label] = label_cnt
                label_cnt += 1 
            data.append(((row.sentence1, row.sentence2), labels[row.gold_label]))
            if cnt == max_lines:
                break
            cnt += 1
    return data, labels

def dump_snli(data, outfile):
    outfile.write('\t'.join(['gold_label','sentence1','sentence2'])+'\n')
    for row in data:
        outfile.write('\t'.join([row[1], row[0][0], str(row[0][1]])))
        outfile.write('\n')

search_exp = re.compile(r'\b(\w+)([\.,!;:])')
replace_exp = r'\1 \2'
def separate_marks(row_nt):
    return row_nt._replace(
        sentence1 = re.sub(search_exp, replace_exp, row_nt.sentence1),
        sentence2 = re.sub(search_exp, replace_exp, row_nt.sentence2)
        )
    