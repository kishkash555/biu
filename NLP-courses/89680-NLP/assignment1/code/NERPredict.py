# hmm greedy
import hmm_main
from os import path
import sys


import MLETrain
base_path = path.dirname(path.abspath(__file__))
ner_path = path.join(base_path,'..','ner')
tagged_input = path.join(ner_path,'train')
qmle = path.join(ner_path,'qmle')
emle = path.join(ner_path,'emle')
blind_test = path.join(ner_path,'test.blind')
blind_test_submit = path.join(ner_path, 'ner.hmm.pred')


if __name__ == "__main__":
    
   # MLETrain.main([sys.argv[0], tagged_input, qmle, emle])
    hmm_main.main([sys.argv[0], blind_test, qmle, emle, blind_test_submit],'greedy')
    