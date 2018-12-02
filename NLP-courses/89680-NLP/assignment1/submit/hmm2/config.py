import numpy as np
import os.path as path

base_path = path.dirname(path.abspath(__file__))

vector_norm = lambda x: x/ np.linalg.norm(x,1)

transition_lambdas = vector_norm(np.array([500,10,1]))
period = "."
start = "*"
stop = "@"

viterbi_prune_keep_count = 9

class frequncies:
    min_word_frequency = 6
    max_word_list_length = 1500
    min_tag_frequency = 1e-3
    min_tag_pair_frequency = 5e-5

class defaultFiles:
    tagged_input = path.join(base_path,'..','data','ass1-tagger-train')
    untagged_test = path.join(base_path,'..','data','ass1-tagger-test-input')
    
    qmle = path.join(base_path,'..','data','qmle')
    emle = path.join(base_path,'..','data','emle')
    hmm_output = path.join(base_path,'..','data','hmm_output')
    
    memm_feature_out = path.join(base_path,'..','data','ass1-tagger-train-sample-out')
    memm_feature_vec = path.join(base_path,'..','data', 'feature_vecs_file')
    memm_feature_map = path.join(base_path,'..','data', 'map_file')
    memm_model_file = path.join(base_path,'..','data', 'memm_model.pickle' )
    memm_greedy_tagged_output = path.join(base_path,'..','data', 'greedy_output' )
    memm_viterbi_tagged_output = path.join(base_path,'..','data', 'viterbi_output' )
    
class featureNamePrefixes:
    is_word = "is_word_"
    prev_tag = "prev_tag_"
