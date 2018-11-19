import numpy as np
import sys
import config
import extractFeatures as ef
from collections import OrderedDict
import scipy.sparse as sp
import pickle
import memm_utils as mu
from viterbi import viterbi_memm

def get_proba(triplet, model, registered_features, feature_dict):
    feature_vec = triplet_to_feature_vec(triplet, registered_features, feature_dict)
    probas = -model.predict_log_proba(feature_vec)
    return probas
        

def generate_greedily_tagged_triplets(sentence_str, model, registered_features, feature_dict, tag_dict):
    tagged_sentence = [(word,'') for word in sentence_str.split()]
    for triplet in ef.generate_triplets_from_tagged_sentence(tagged_sentence):
        #triplet = triplet[1:] + [(word,'')]
        probas = get_proba(triplet, model, registered_features, feature_dict)
        greedy_tag_ord = np.argmin(probas)
        greedy_tag_name = tag_dict[greedy_tag_ord]
        triplet[2] = (triplet[2][0],greedy_tag_name)
        yield triplet[2]

def triplet_to_feature_vec(triplet, registered_features, feature_dict):
    feature_value_pairs = sum(ef.triplet_to_feature_name_value_pairs(triplet,registered_features), [])
    feature_names = [feat.split('=')[0] for feat in feature_value_pairs]
    indices = np.array([feature_dict[name] for name in feature_names if name in feature_dict])
    vec_len = max(feature_dict.values())+1
    csr = (np.ones(len(indices),np.double), (np.zeros(len(indices),np.int), indices))
    feature_vec = sp.csr_matrix(csr, shape=(1,vec_len))
    return feature_vec

def viterbi_tagged_triplets(sentence_str, model, registered_features, feature_dict, tag_dict):
    def scoring_func(triplet):
        return get_proba(triplet, model, registered_features, feature_dict)
    y = viterbi_memm(sentence_str, [tag_dict[i] for i in range(len(tag_dict))], scoring_func)
    return zip(sentence_str.split(),y)

def triplets_to_tagged_line(tagged_triplets):
    tagged_line = ' '.join(["/".join(t) for t in tagged_triplets])
    return tagged_line

def main(argv, decoder):
    decoder = decoder.lower()
    if decoder == 'greedy':
        decoder_func = generate_greedily_tagged_triplets
    elif decoder == 'viterbi':
        decoder_func = viterbi_tagged_triplets
    else:
        print(f"decoder should be 'greedy' or 'viterbi', not {decoder}")
        return
    if len(argv)==1:
        print("Using default file names")
        untagged_file = config.defaultFiles.untagged_test
        model_file = config.defaultFiles.memm_model_file
        feature_map_file = config.defaultFiles.memm_feature_map
        tagged_out_file = config.defaultFiles.memm_greedy_tagged_output if decoder == 'greedy' else config.defaultFiles.memm_viterbi_tagged_output
    elif len(sys.argv) != 5 :
        print(f"usage: {sys.argv[0]} input_file model_file feature_map_file out_file\nexiting.")
        exit()
    else:
        untagged_file = sys.argv[1]
        model_file = sys.argv[2]
        feature_map_file = sys.argv[3]
        tagged_out_file = sys.argv[4]
    

    
    tag_dict, feature_dict = mu.load_map_file(feature_map_file)
    model = pickle.load(open(model_file,'rb'))
    
    funcs_with_offsets = [ef.vocab_predicates(set(),i) for i in range(5)]
    is_word_funcs = [f[0] for f in funcs_with_offsets]
    registered_features = ef.registered_features + is_word_funcs + list(funcs_with_offsets[2][1:])
    
    c = 0
    with open(tagged_out_file,'wt',encoding='utf8') as o:
        with open(untagged_file,'rt',encoding='utf8') as i:
            for line in i:
                tagged_triplets = decoder_func(line, model, registered_features, feature_dict, tag_dict)   
                o.write(triplets_to_tagged_line(tagged_triplets)+'\n')
                c += 1
                if c % 1 == 0:
                    print(f'wrote {c} lines.')
    print('Done.')



