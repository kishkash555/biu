#!/usr/bin/python2
import dynet as dy
import datetime
from collections import Counter, OrderedDict
from os import path
import numpy as np
import pickle
import sys
import time

EMB = 50
INPUT = EMB*5
INITIAL_LEARN_RATE = 0.15 # can be changed by argv[1]
HIDDEN = 400  # can be changed by argv[2]
SET_RANDOM_SEED = True # can be changed by argv[3]
global MIN_ACC 
MIN_ACC = 0.80
INPUT_DIR = 'pos' # can be changed by argv[4]
REG_LAMBDA = 0.02 # can be changed by argv[5]

DICTS_FILE = 'dicts.pickle'
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


def create_network_params(nwords, ntags):
    # create a parameter collection and add the parameters.
    print("adding parameters")
    m = dy.ParameterCollection()
    E = m.add_lookup_parameters((nwords,EMB), name='E')
    b = m.add_parameters(HIDDEN, name='b')
    U = m.add_parameters((ntags, HIDDEN), name='U')
    W = m.add_parameters((HIDDEN, INPUT), name='W')
    bp = m.add_parameters(ntags, name='bp')
    dy.renew_cg()
    return m, E, b, U, W, bp

def build_network(params, x_ordinals):
    _, E, b, U, W, bp = params
    x = dy.concatenate([E[ord] for ord in x_ordinals])
    output = dy.softmax(U * (dy.tanh(W*x + b)) + bp)
    return output

def train_network(params, ntags, train_data, dev_set):
    global telemetry_file, randstring, MIN_ACC
    prev_acc = 0
    m = params[0]
    t0 = time.clock()
    # train the network
    trainer = dy.SimpleSGDTrainer(m)
    total_loss = 0
    seen_instances = 0
    train_good = 0
    for train_x, train_y in train_data:
        dy.renew_cg()
        output = build_network(params, train_x)
        # l2 regularization did not look promising at all, so it's commented out
        loss = -dy.log(output[train_y])  + REG_LAMBDA * sum([dy.l2_norm(p) for p in params[2:]])
        if train_y == np.argmax(output.npvalue()):
            train_good +=1
        seen_instances += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        
        if seen_instances % 20000 == 0:
            # measure elapsed seconds
            secs = time.clock() - t0
            t0 = time.clock()
            good = case = 0
            max_dev_instances = 70*1000
            dev_instances = 0
            for x_tuple, dev_y in dev_set:
                output =  build_network(params, x_tuple)
                if np.argmax(output.npvalue()) == dev_y:
                    good +=1
                case +=1
                dev_instances += 1
                if dev_instances >= max_dev_instances:
                    break
            acc = float(good)/case
            print("iterations: {}. train_accuracy: {} accuracy: {} avg loss: {} secs per 1000:{}".format(seen_instances, float(train_good)/20000, acc, total_loss / (seen_instances+1), secs/20))
            train_good = 0
            if acc > MIN_ACC and acc > prev_acc:
                print("saving.")
                dy.save("params_"+randstring,list(params)[1:])
                prev_acc = acc
            
            telemetry_file.write("{}\t{}\t{}\t{}\n".format(seen_instances, acc, total_loss / (seen_instances+1), secs/20))
    MIN_ACC = max(prev_acc, MIN_ACC) 


def scan_train_for_vocab(train_data):
    words = Counter()
    tags = Counter()
    for line in train_data:
        if len(line) > 1:
            word, tag = line.split()
            words[word] += 1
            tags[tag] += 1
        
    word_list = [a for a, _ in words.most_common()]
    tag_list = [a for a, _ in tags.most_common()]
    word_dict = OrderedDict((a,i) for i, a in enumerate(word_list))
    tag_dict = OrderedDict((a,i) for i, a in enumerate(tag_list))
    return word_dict, tag_dict


def train_stream_to_sentence_tuples(input_file):
    sentence = [('**START**', '')]*2
    for line in input_file:
        if len(line)>1:
            word, tag = line.split()
            sentence.append((word,tag))
        elif len(sentence) > 2:
            sentence += [('**STOP**', '')]*2
            yield sentence
            sentence = [('**START**', '')]*2


def generate_train_5tuples(tagged_sentence_stream, word_dict, tag_dict, unk_threshold):
    """
    generate a 5-tuple of indices
    and a y one-hot vector
    based on the current word + 2 words of context from each side
    """
    for tagged_sentence in tagged_sentence_stream:
        train_x_tuple = []
        train_y_tuple = []
        for word, tag in tagged_sentence:
            if word in word_dict:
                train_x_tuple.append(word_dict[word])
            else: 
                train_x_tuple.append(word_dict['**UNK**'])
            train_y_tuple.append(tag)
            if len(train_x_tuple) == 5:
                yield list(train_x_tuple), tag_dict[train_y_tuple[2]]
                
                train_x_tuple.pop(0)
                train_y_tuple.pop(0)
                


if __name__ == "__main__":
    global telemetry_file, randstring
    argv = sys.argv
    randstring = str(datetime.datetime.now().microsecond)
    telemetry_file= open('telem'+randstring+'.txt','wt')

    if len(argv)>1 and argv[1] != "--":
        INITIAL_LEARN_RATE = float(argv[1])
    if len(argv)>2 and argv[2] != "--":
        HIDDEN = int(argv[2]) 
    if len(argv)>3 and argv[3] != "--":
        SET_RANDOM_SEED = bool(int(argv[3]))
    if len(argv)>4 and argv[4] != "--":
        INPUT_DIR = argv[4]
    if len(argv)>5 and argv[5] != "--":
        REG_LAMBDA = float(argv[5])
    if SET_RANDOM_SEED:
        dyparams =  dy.DynetParams()
        dyparams.set_random_seed(1234)
        dyparams.init()
        np.random.seed(54321)

    print("input: {}\nlearning rate: {}\nhidden layer size: {}\nrandom: {}, lambda: {}".format(INPUT_DIR,INITIAL_LEARN_RATE,HIDDEN,SET_RANDOM_SEED,REG_LAMBDA))
    telemetry_file.write("{} {} {} {} {} {}\n".format(argv[0],INITIAL_LEARN_RATE,HIDDEN,int(SET_RANDOM_SEED),INPUT_DIR, REG_LAMBDA))
    telemetry_file.write("iterations\taccuracy\tavg_loss\tsecs_per_1000\n")

    input_file = path.join('..',INPUT_DIR,'train')
    dev_file = path.join('..',INPUT_DIR,'dev')
    with open(input_file,'rt') as i:
        word_dict, tag_dict = scan_train_for_vocab(i)
    word_dict['**START**'] = len(word_dict)
    word_dict['**STOP**'] = len(word_dict)
    word_dict['**UNK**'] = len(word_dict)
    tag_dict[''] = len(tag_dict)
    
    # save the dictionaries
    with open(DICTS_FILE,'wb') as f:
        pickle.dump( { 'word_dict': word_dict, 'tag_dict': tag_dict}, f)

    params = create_network_params(len(word_dict), len(tag_dict))

    with open(dev_file, 'rt') as d:
        dev_set = list(generate_train_5tuples(train_stream_to_sentence_tuples(d), word_dict, tag_dict, 0))

    if 'O' in tag_dict:
        dev_set_len = len(dev_set)
        dev_set = [tup for tup in dev_set if tup[1] != tag_dict['O']]
        print("Removing 'O' from dev set. size before: {} size after: {}".format(dev_set_len, len(dev_set)))
        
    # print(list(generate_train_tuples(dev_set[:10],word_dict, tag_dict)))
    # print(tag_dict)
    unk_threshold = int(len(word_dict)/10) + 1
    ntags = len(tag_dict)
    with open(input_file,'rt') as i:
        train_data = list(generate_train_5tuples(train_stream_to_sentence_tuples(i), word_dict, tag_dict, unk_threshold))

    try:
        for i in range(10):
            np.random.shuffle(train_data)
            print(train_data[:5])
            train_network(params, ntags, train_data, dev_set)
    finally:
        print("INTERRUPTED. closing file")
        telemetry_file.close() 