#!/usr/bin/python3
import dynet as dy
from collections import Counter, OrderedDict
from os import path
import numpy as np


## y = softmax(U(tanh(Ex+b))+b')
def build_ex1_network(nwords,ntags):

    # create a parameter collection and add the parameters.
    print("adding parameters")
    m = dy.ParameterCollection()
    E = m.add_parameters((50,nwords))
    b = m.add_parameters((50))
    U = m.add_parameters((ntags,50))
    bp = m.add_parameters((ntags))
    dy.renew_cg()

    x = dy.vecInput(nwords)
    output = dy.softmax( U * (dy.tanh(E * x + b)) + bp) 
    return x, m, E, output

def train_network(x, m, E, network_output, train_data):
    first_x, first_y = next(train_data)
    print("first x {} first y {}".format(first_x, first_y))
    x.set(first_x)
    y = dy.vecInput(first_y.shape[0])
    y.set(first_y)
    loss = dy.binary_log_loss(network_output, y)

    # train the network
    trainer = dy.SimpleSGDTrainer(m)

    total_loss = 0
    seen_instances = 0
    for train_x, train_y in train_data:
        x.set(train_x)
        y.set(train_y)
        seen_instances += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        if seen_instances % 100 == 0:
            print("average loss is:",total_loss / seen_instances)
            print("E is: {}".format(E.value()))
    return 


# function for graph creation
# def create_network_return_loss(inputs, expected_output):
#     """
#     inputs is a list of numbers
#     """
#     dy.renew_cg()
#     emb_vectors = [lookup[i] for i in inputs]
#     net_input = dy.concatenate(emb_vectors)
#     net_output = dy.softmax( (pW*net_input) + pB)
#     loss = -dy.log(dy.pick(net_output, expected_output))
#     return loss

#     loss = dy.binary_log_loss()

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

def generate_tagged_sentences(input_file):
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
    for tagged_sentence in tagged_sentences:
        for word_oh, tag_oh in tagged_sentence_to_train_data(tagged_sentence, word_dict, tag_dict):
            yield word_oh, tag_oh


def tagged_sentence_to_train_data(tagged_sentence,word_dict, tag_dict):
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
    with open(input_file,'rt',encoding='utf8') as i:
        word_dict, tag_dict = scan_train_for_vocab(i)
    word_dict['**START**'] = len(word_dict)
    word_dict['**STOP**'] = len(word_dict)
    tag_dict[''] = len(tag_dict)
    
    x, m, E, output = build_ex1_network(len(word_dict), len(tag_dict))

    with open(input_file,'rt',encoding='utf8') as i:
        train_data_iter = generate_train_data(generate_tagged_sentences(i), word_dict, tag_dict)

        train_network(x, m, E, output, train_data_iter)
  
  
        # for tagged_word in sentence:
        #         train_network
        #     tagged_sentence_to_train_data
        #     codes_sentence = [(word_dict[a],tag_dict[b]) for a,b in sentence]
        #     print(sentence)
        #     print(codes_sentence)
            
        #     c += 1
        #     if c==5: 
        #         break
            