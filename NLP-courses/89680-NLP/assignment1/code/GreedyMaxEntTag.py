import numpy as np
import sys
import config
import extractFeatures as ef
from collections import OrderedDict
import scipy.sparse as sp
import pickle

def load_map_file(feature_map_file):
    with open(feature_map_file,'rt',encoding='utf8') as mf:
        tag_dict = dict()
        feature_dict = dict()
        for line in mf:
            name, ind = line.split()
            name, value = name.split('=')
            if name=='tag':
                tag_dict[int(ind)] = value 
            else:
                feature_dict[name] = int(ind)  
            
    return tag_dict, feature_dict

def generate_greedily_tagged_triplets(sentence_str, model, registered_features, feature_dict, tag_dict):
    sentence = sentence_str.split()
    triplet = [('',config.start)]*3
    for word in sentence:
        triplet = triplet[1:] + [(word,'')]
        feature_vec = triplet_to_feature_vec(triplet, registered_features, feature_dict)
        probas = -model.predict_log_proba(feature_vec)
        greedy_tag_ord = np.argmin(probas)
        greedy_tag_name = tag_dict[greedy_tag_ord]
        triplet[2] = (word,greedy_tag_name)
        yield triplet[2]

def triplet_to_feature_vec(triplet, registered_features, feature_dict):
    feature_value_pairs = sum(ef.triplet_to_feature_name_value_pairs(triplet,registered_features), [])
    indices = np.array([feature_dict[feat.split('=')[0]] for feat in feature_value_pairs])
    vec_len = max(feature_dict.values())+1
    csr = (np.ones(len(indices),np.double), (np.zeros(len(indices),np.int), indices))
    feature_vec = sp.csr_matrix(csr, shape=(1,vec_len))
    return feature_vec
    

def get_registered_features(feature_names):
    registered_words = []
    registered_tags = []
    registered_tag_pairs = []
    is_word = "is_word_"
    iwl = len(is_word)
    prev_tag= "prev_tag_"
    ptl = len(prev_tag)
    prev_2 = "last_2_tags_"
    p2l = len(prev_2)
    for name in feature_names:
        if name.startswith(is_word):
            registered_words.append(name[iwl:])
        elif name.startswith(prev_tag):
            registered_tags.append(name[ptl:])
        elif name.startswith(prev_2):
            tag1,tag2 = name[p2l:].split("_")
            registered_tag_pairs.append((tag1,tag2))
    return registered_words, registered_tags, registered_tag_pairs


def triplets_to_tagged_line(tagged_triplets):
    tagged_line = ' '.join(["/".join(t) for t in tagged_triplets])
    return tagged_line


if __name__ == "__main__":
    if len(sys.argv)==1:
        print("Using default file names")
        untagged_file = config.defaultFiles.untagged_test
        model_file = config.defaultFiles.memm_model_file
        feature_map_file = config.defaultFiles.memm_feature_map
        tagged_out_file = config.defaultFiles.memm_greedy_tagged_output
    elif len(sys.argv) != 5 :
        print(f"usage: {sys.argv[0]} feature_file model_file\nexiting.")
        exit()
    else:
        untagged_file = sys.argv[1]
        model_file = sys.argv[2]
        feature_map_file = sys.argv[3]
        tagged_out_file = sys.argv[4]
    
    tag_dict, feature_dict = load_map_file(feature_map_file)
    model = pickle.load(open(model_file,'rb'))
    registered_words, registered_tags, registered_tag_pairs = get_registered_features(feature_dict.keys())
    registered_features = ef.registered_features +\
         [ef.is_word(w) for w in registered_words] + \
         [ef.prev_tag(t) for t in registered_tags] + \
         [ef.previous_2_tags(*tp) for tp in registered_tag_pairs]

    c = 0
    with open(tagged_out_file,'wt',encoding='utf8') as o:
        with open(untagged_file,'rt',encoding='utf8') as i:
            for line in i:
                c += 1
                tagged_triplets = generate_greedily_tagged_triplets(line, model, registered_features, feature_dict, tag_dict)   
                o.write(triplets_to_tagged_line(tagged_triplets)+'\n')
                if c % 1000 == 0:
                    print(f'wrote {c} lines.')
    print('Done.')

