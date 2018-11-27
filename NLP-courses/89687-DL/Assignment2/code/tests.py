#!/usr/bin/python2
from os import path
import WordBindings as wb
input_file = path.join('..','pos','train')

with open(input_file,'rt') as i:
        word_dict, tag_dict = wb.scan_train_for_vocab(i)

word_dict['**START**'] = len(word_dict)
word_dict['**STOP**'] = len(word_dict)
tag_dict[''] = len(tag_dict)

with open(input_file,'rt') as i:
    a = wb.generate_tagged_sentences(i)
    c = 0
    for s in a:
        c+=1
    print(c)

with open(input_file,'rt') as i:
    a = wb.generate_train_tuples(wb.generate_tagged_sentences(i),word_dict, tag_dict)
    c = 0
    for s in a:
        c+=1
    print(c)
