#!/usr/bin/python2
import dynet as dy
from collections import Counter, OrderedDict
from os import path
import numpy as np
import pickle


EMB = 50
INPUT = EMB*5
HIDDEN = 400

def create_network_params(nwords, ntags):
    # create a parameter collection and add the parameters.
    print("adding parameters")
    m = dy.ParameterCollection()
    E = m.add_lookup_parameters((nwords,EMB))
    b = m.add_parameters(HIDDEN)
    U = m.add_parameters((ntags, HIDDEN))
    W = m.add_parameters((HIDDEN, INPUT))
    bp = m.add_parameters(ntags)
    dy.renew_cg()
    # x = dy.vecInput(250)
    return m, E, b, U, W, bp

def build_network(params, x_ordinals):
    m, E, b, U, W, bp = params
    x = dy.concatenate([E[ord] for ord in x_ordinals])
    output = dy.softmax(U * (dy.tanh(W*x + b)) + bp)
    return output

def train_network(params, ntags, train_data, dev_set):
    # m, E, b, U, W, bp = params
    m = params[0]
    
    # train the network
    trainer = dy.SimpleSGDTrainer(m)
    total_loss = 0
    seen_instances = 0
    for train_x, train_y in train_data:
        dy.renew_cg()
        output = build_network(params, train_x)
        loss = -dy.log(output[train_y])

        seen_instances += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        
        if seen_instances % 20000 == 0:
            print("average loss after {} iterations: {}. current loss: {}".format(seen_instances, total_loss / seen_instances, loss.value()))
            good = case = 0
            max_dev_instances = 50*1000
            dev_instances = 0
            for x_tuple, dev_y in dev_set:
                #dy.renew_cg()
                output =  build_network(params, x_tuple)
                if np.argmax(output.npvalue()) == dev_y:
                    good +=1
                case +=1
                dev_instances += 1
                if dev_instances >= max_dev_instances:
                    break
            print("accuracy after {} iterations: {}".format(seen_instances, float(good)/case))
    return 


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


def one_hot(k, n):
    ret = np.zeros(n)
    ret[k]=1.
    return ret

def generate_train_data(tagged_sentences, word_dict, tag_dict):
    """
    generates an x one-hot vector and a y one-hot vector 
    based on the current word and its tag 
    """
    for tagged_sentence in tagged_sentences:
        for word_oh, tag_oh in tagged_sentence_to_train_data(tagged_sentence, word_dict, tag_dict):
            yield word_oh, tag_oh


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
                



def tagged_sentence_to_train_data(tagged_sentence, word_dict, tag_dict):
    """
    for now, does not implement embeddings
    """
    nwords = len(word_dict)
    ntags = len(tag_dict)
    
    for word, tag in tagged_sentence[2:-2]:
        word_oh = one_hot(word_dict[word],nwords)
        tag_oh = one_hot(tag_dict[tag],ntags)
        #print("word_oh: {} tag_oh: {}".format(type(word_oh), type(tag_oh)))
        yield word_oh, tag_oh


if __name__ == "__main__":
    input_file = path.join('..','pos','train')
    dev_file = path.join('..','pos','dev')
    with open(input_file,'rt') as i:
        word_dict, tag_dict = scan_train_for_vocab(i)
    word_dict['**START**'] = len(word_dict)
    word_dict['**STOP**'] = len(word_dict)
    word_dict['**UNK**'] = len(word_dict)
    tag_dict[''] = len(tag_dict)
    

    params = create_network_params(len(word_dict), len(tag_dict))

    with open(dev_file, 'rt') as d:
        dev_set = list(generate_train_5tuples(train_stream_to_sentence_tuples(d), word_dict, tag_dict, 0))


    # print(list(generate_train_tuples(dev_set[:10],word_dict, tag_dict)))
    # print(tag_dict)
    unk_threshold = int(len(word_dict)/10) + 1
    ntags = len(tag_dict)
    with open(input_file,'rt') as i:
        train_data = list(generate_train_5tuples(train_stream_to_sentence_tuples(i), word_dict, tag_dict, unk_threshold))

    print(train_data[:5])
    train_network(params, ntags, train_data, dev_set)
  
            