#!/usr/bin/python2

import datetime
from collections import Counter, OrderedDict
from os import path
import numpy as np
import pickle
import sys
import time
import mlp
import _dynet as dy

INPUT_DIR = 'pos' # can be changed by argv[4]

USE_EXT_EMBEDDINGS = False # can be changed by argv[6]
EMBED_PREFIX_AND_SUFFIX = False # can be changed by argv[7]
SET_RANDOM_SEED = True
DICTS_FILE = 'dicts_pickle'
doc = \
"""
Checklist of main points implemented:
    1. build the expression dy.softmax(U * (dy.tanh(W*x + b)) + bp) with x constructed for each train example from an embedding matrix.
    2. train the network. Things to note during training:
        2a. measure accuracy on dev every 20,000 iterations. record the accuracy progression to a "telemetry" file
        2b. remove 'O' from dev set (create bias in accuracy measure)
        2c. shuffle the input order randomly
        2d. reprocess the training examples (up to 10 times if the user doesn't hit ^C earlier)
        2e. save the network parameters to a file if results are good on dev set
    3. get the following inputs from the command line:
        3a. initial learning rate
        3b. size of hidden layer
        3c. wheter to use a fixed random see
        3d. the source directory for the "train" and "dev" files
    4. every word that is in the bottom decile in terms of frequency will be also learned as **UNK**
    5. every word encountered during tagging that is not in the trained vocabulary gets replaced with **UNK**
"""

def is_a_number(word):
    return len(set(word) - set('-+,.0123456789'))==0

def create_word_and_tag_dict(train_file, embedding_vocab, rare_word_threshold, add_prefix_suffix):
    """
    construct dictionary of known words with their location.
    due to usage of OrderedDict two-way conversion is possible:
    - word_dict[word] = location 
    - list(word_dict.keys())[location]==word 
    """
    ext_word_dict = OrderedDict([(mlp.UNK if a=='UUUNKKK' else a,i) for i, a in enumerate(embedding_vocab)])
    
    train_word_counter, tag_dict = scan_train_for_vocab(train_file)
    rare_word_set = set()
    
    i = len(ext_word_dict)
    for word, count in train_word_counter.most_common():
        if word not in ext_word_dict and not is_a_number(word):
            ext_word_dict[word] = i
            i += 1
            if add_prefix_suffix:
                pre = mlp.PREFIX_MARK + word_pre(word)
                suf = word_suff(word) + mlp.SUFFIX_MARK
                if pre not in ext_word_dict:
                    ext_word_dict[pre] = i
                    #print("adding {} for word {}".format(word_pre(word),word))
                    i += 1
                if suf not in ext_word_dict:
                    ext_word_dict[suf] = i
                    i += 1
                
        if count <= rare_word_threshold and not is_a_number(word):
            rare_word_set.add(word)

    ext_word_dict[mlp.START] = i 
    ext_word_dict[mlp.STOP] = i + 1
    ext_word_dict[mlp.NUMBER] = i + 2
    if mlp.UNK not in ext_word_dict:
        ext_word_dict[mlp.UNK] = i + 3
    tag_dict[''] = len(tag_dict)
    print("size ext_word_dict {}".format(len(ext_word_dict)))
    return ext_word_dict, tag_dict, rare_word_set

def scan_train_for_vocab(train_data):
    words = Counter()
    tags = Counter()
    for line in train_data:
        if len(line) > 1:
            word, tag = line.split()
            words[word] += 1
            tags[tag] += 1
        
    tag_list = [a for a, _ in tags.most_common()]
    tag_dict = OrderedDict((a,i) for i, a in enumerate(tag_list))
    return words, tag_dict


def train_stream_to_sentence_tuples(input_file):
    sentence = [(mlp.START, '')] * 2
    for line in input_file:
        if len(line)>1:
            word, tag = line.split()
            sentence.append((word,tag))
        elif len(sentence) > 2:
            sentence += [(mlp.STOP, '')] * 2
            yield sentence
            sentence = [(mlp.START, '')] * 2

def generate_train_5tuples_with_prefix_suffix(tagged_sentences, word_dict, tag_dict, rare_word_set): 
    stream1 = generate_train_5tuples(tagged_sentences, word_dict, tag_dict, rare_word_set)
    stream2 = generate_train_5tuples(convert_tagged_sentence_to_prefixes(tagged_sentences), word_dict, tag_dict, set())
    stream3 = generate_train_5tuples(convert_tagged_sentence_to_suffixes(tagged_sentences), word_dict, tag_dict, set())
    for a, b, c in zip(stream1, stream2, stream3):
        x =  {"fullwords": a[0], "prefix": b[0], "suffix": c[0]}
        #print(x)
        yield x, a[1] 
    

def generate_train_5tuples(tagged_sentence_stream, word_dict, tag_dict, rare_word_set):
    """
    generate a 5-tuple of indices
    and a y one-hot vector
    based on the current word + 2 words of context from each side
    """
    for tagged_sentence in tagged_sentence_stream:
        train_x_tuple = []
        train_y_tuple = []
        for word, tag in tagged_sentence:
            if is_a_number(word): # a number
                train_x_tuple.append(word_dict[mlp.NUMBER])
            elif word in word_dict:
                train_x_tuple.append(word_dict[word])
            elif word.lower() in word_dict: 
                train_x_tuple.append(word_dict[word.lower()])
            elif '-' in word and word.split('-')[-1] in word_dict:
                train_x_tuple.append(word_dict[word.split('-')[-1]])
            else:
                train_x_tuple.append(word_dict[mlp.UNK])
            train_y_tuple.append(tag)
            if len(train_x_tuple) == 5:
                yield list(train_x_tuple), tag_dict[train_y_tuple[2]]
                if word in rare_word_set:
                    train_x_tuple[2] = word_dict[mlp.UNK]
                    yield list(train_x_tuple), tag_dict[train_y_tuple[2]]
                   
                train_x_tuple.pop(0)
                train_y_tuple.pop(0)

word_pre =  lambda(word): word[:3 ] if len(word) >= mlp.MIN_LENGTH_FOR_PRE_SUF and word not in { mlp.UNK, mlp.START, mlp.STOP, mlp.NUMBER} else ''
word_suff = lambda(word): word[-3:] if len(word) >= mlp.MIN_LENGTH_FOR_PRE_SUF and word not in { mlp.UNK, mlp.START, mlp.STOP, mlp.NUMBER} else ''

def convert_tagged_sentence_to_prefixes(tagged_sentences):
    for tagged_sentence in tagged_sentences:
        prefix_sentence = [( mlp.PREFIX_MARK + word_pre(word), tag) for word, tag in tagged_sentence]
        prefix_sentence
        yield prefix_sentence

def convert_tagged_sentence_to_suffixes(tagged_sentences):
    for tagged_sentence in tagged_sentences:
        suffix_sentence = [( word_suff(word) + mlp.SUFFIX_MARK, tag) for word, tag in tagged_sentence]
        yield suffix_sentence
        

    

if __name__ == "__main__":
    global telemetry_file, randstring
    argv = sys.argv
    randstring = str(datetime.datetime.now().microsecond)
    telemetry_file= open('telem'+randstring+'.txt','wt')

    if len(argv)>1 and argv[1] != "--":
        try:
            mlp.INITIAL_LEARN_RATE = float(argv[1])
        except:
            print(['usage: {} [PARAM1|[-- PARAM2|[-- ... [PARAM6]]]\nPARAM NAMES AND DEFAULTS:\n'
            '\tPARAM1: INITIAL LEARNING RATE [{}]\n'
            '\tPARAM2: HIDDEN LAYER SIZE [{}]\n'
            '\tPARAM3: PRESET RANDOM SEED 1=True 0=False [{}]\n'
            '\tPARAM4: INPUT DIR [{}]\n'
            '\tPARAM5: not in use\n'
            #'\tPARAM5: REG LAMBDA [{}]\n'
            '\tPARAM6: USE EXTERNAL EMBEDDINGS 1=True 0=False.\n\t\tLooks for ../wordVectors.txt and ../vocab.txt [{}]\n'
            '\tPARAM7: EMBED PREFIX AND SUFFIX 1=True 0=Fasle. [{}]'
            ][0].format(argv[0],mlp.INITIAL_LEARN_RATE, mlp.HIDDEN, SET_RANDOM_SEED, INPUT_DIR, USE_EXT_EMBEDDINGS, EMBED_PREFIX_AND_SUFFIX))
            raise ValueError()
    if len(argv)>2 and argv[2] != "--":
        mlp.HIDDEN = int(argv[2]) 
    if len(argv)>3 and argv[3] != "--":
        SET_RANDOM_SEED = bool(int(argv[3]))
    if len(argv)>4 and argv[4] != "--":
        INPUT_DIR = argv[4]
    if INPUT_DIR == 'pos':
        mlp.MIN_ACC = 0.94
    if len(argv)>5 and argv[5] != "--": # not in use
        mlp.REG_LAMBDA = float(argv[5])
    if len(argv)>6 and argv[6] != "--":
        USE_EXT_EMBEDDINGS = bool(int(argv[6]))
    if len(argv)>7 and argv[7] != "--":
        EMBED_PREFIX_AND_SUFFIX = bool(int(argv[7]))
    dyparams =  dy.DynetParams()
    if SET_RANDOM_SEED:
        dyparams.set_random_seed(1234)
        np.random.seed(54321)
    dyparams.init()
        
    
    print("input: {}\nlearning rate: {}\nhidden layer size: {}\nrandom: {}, INPUT_DIR: {}\nExt embediings: {}, prefix suffix: {}"\
        .format(INPUT_DIR, mlp.INITIAL_LEARN_RATE, mlp.HIDDEN, SET_RANDOM_SEED, INPUT_DIR, USE_EXT_EMBEDDINGS, EMBED_PREFIX_AND_SUFFIX))
    telemetry_file.write("{} {} {} {} {} {} {} {}\n".format(argv[0], mlp.INITIAL_LEARN_RATE, mlp.HIDDEN, int(SET_RANDOM_SEED), INPUT_DIR, mlp.REG_LAMBDA, USE_EXT_EMBEDDINGS, EMBED_PREFIX_AND_SUFFIX))
    telemetry_file.write("iterations\taccuracy\tavg_loss\tsecs_per_1000\n")

    input_file = path.join('..',INPUT_DIR,'train')
    dev_file = path.join('..',INPUT_DIR,'dev')

    
    if USE_EXT_EMBEDDINGS:
        provided_emb_matrix = np.fromfile('../wordVectors.txt', sep=' ').reshape(100232, 50)
        with open('../vocab.txt','rt') as row:
           provided_emb_vocab = [w.strip() for w in row]
        print('some 5 words: {}\nfirst 3 values of first 5 vectors: {}'.format(provided_emb_vocab[100:105], provided_emb_matrix[:5,:3]))
    else:
        provided_emb_matrix = np.array([],float)
        provided_emb_vocab = []
    
    with open(input_file,'rt') as i:
        word_dict, tag_dict, rare_word_set = create_word_and_tag_dict(i, provided_emb_vocab, 1, EMBED_PREFIX_AND_SUFFIX)

    rare_word_set = list(rare_word_set)
    np.random.shuffle(rare_word_set)
    rare_word_set = set(rare_word_set[:4000])
    print("rare words count: {} of {}".format(len(rare_word_set), len(word_dict)))
    print("rare word examples: {}".format(" ".join(list(rare_word_set)[::50])))
    # save the dictionaries
    with open(DICTS_FILE + '.' + INPUT_DIR ,'wb') as f:
        pickle.dump( { 'word_dict': word_dict, 'tag_dict': tag_dict}, f)

    params = mlp.create_network_params(len(word_dict), len(tag_dict))

    # dev set: loading
    with open(dev_file, 'rt') as d:
        dev_set = list(generate_train_5tuples(train_stream_to_sentence_tuples(d), word_dict, tag_dict, set()))
    # dev set: clearing 'O' tag (not counted towards accuracy)
    if 'O' in tag_dict:
        dev_set_len = len(dev_set)
        dev_set = [tup for tup in dev_set if tup[1] != tag_dict['O']]
        print("Removing 'O' from dev set. size before: {} size after: {}".format(dev_set_len, len(dev_set)))
        
    ntags = len(tag_dict)
    with open(input_file,'rt') as i:
        input_sentences = list(train_stream_to_sentence_tuples(i))
    
    try:
        for i in range(10):
            np.random.shuffle(input_sentences)
            #print(input_sentences[:5])
            
            if EMBED_PREFIX_AND_SUFFIX:
                train_data = generate_train_5tuples_with_prefix_suffix(input_sentences, word_dict, tag_dict, rare_word_set)
            else:
                train_data = generate_train_5tuples(input_sentences, word_dict, tag_dict, rare_word_set)
                print(next(train_data))
            mlp.train_network(params, ntags, train_data, dev_set,telemetry_file, randstring)
    except KeyboardInterrupt:
        print("INTERRUPTED.")
    finally:
        print("closing file")
        telemetry_file.close() 

