#!/usr/bin/python2
import dynet as dy
from collections import Counter, OrderedDict
from os import path
import numpy as np
import pickle


## y = softmax(U(tanh(Ex+b))+b')
def build_ex1_network(nwords, ntags):
    # create a parameter collection and add the parameters.
    print("adding parameters")
    m = dy.ParameterCollection()
    E = m.add_lookup_parameters((nwords,50))
    b = m.add_parameters((250))
    U = m.add_parameters((ntags,250))
    bp = m.add_parameters((ntags))
    dy.renew_cg()
    x = dy.vecInput(250)
    output = dy.softmax( U * (dy.tanh(x + b)) + bp) 
    return x, m, E, b, U, bp

def train_network(x, m, E, b, U, bp, train_data, dev_set):
    first_x, first_y = next(train_data)
    print("first x {} first y {}".format(first_x, first_y))
    x.set(first_x)
    ntags = bp.npvalue().shape[0]
    print("*ntags: {}".format(ntags))
    y = dy.vecInput(ntags)
    y.set(one_hot(first_y,ntags))
    
    # train the network
    trainer = dy.SimpleSGDTrainer(m)
    total_loss = 0
    seen_instances = 0
    for train_x, train_y in train_data:
        # print(train_x, train_y)
        dy.renew_cg()
        a = train_x
        e0 = E[a[0]]
        e1 = E[a[1]]
        e2 = E[a[2]]
        e3 = E[a[3]]
        e4 = E[a[4]]
        e05 = dy.concatenate([e0, e1, e2, e3, e4])
        output = dy.softmax( U * (dy.tanh(e05 + b)) + bp)
        y.set(one_hot(train_y,ntags))
        loss = -dy.log(output[train_y])

        seen_instances += 1
        total_loss += loss.value()
        loss.backward()
        trainer.update()
        
        if seen_instances % 1000 == 0:
            print("average loss after {} iterations: {}. current loss: {}".format(seen_instances, total_loss / seen_instances, loss.value()))
            with open("ex1_embed_mat.pickle",'wb') as f:
                pickle.dump(E.value(),f)
            #print("E is: {}".format(E.value()))
            good = case = 0
            for x_tuple, dev_y in list(generate_train_tuples(dev_set,word_dict, tag_dict))[:50]:
                dy.renew_cg()
                e = [E[x] for x in x_tuple]
                e5 = dy.concatenate(e)
                output = dy.softmax( U * (dy.tanh(e5 + b)) + bp)
                #print(x_tuple, output.npvalue(), dev_y)
                if np.argmax(output.npvalue()) == dev_y:
                    good +=1
                case +=1
            print("accuracy after {} iterations: {}".format(seen_instances, float(good)/case))
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
    """
    generates an x one-hot vector and a y one-hot vector 
    based on the current word and its tag 
    """
    for tagged_sentence in tagged_sentences:
        for word_oh, tag_oh in tagged_sentence_to_train_data(tagged_sentence, word_dict, tag_dict):
            yield word_oh, tag_oh


def generate_train_tuples(tagged_sentences, word_dict, tag_dict):
    """
    generate a 5-tuple of indices
    and a y one-hot vector
    based on the current word + 2 words of context from each side
    """
    for tagged_sentence in tagged_sentences:
        train_x_tuple = []
        train_y_tuple = []
        for word, tag in tagged_sentence:
            if word in word_dict:
                train_x_tuple.append(word_dict[word])
            #else: 
            #    train_x_tuple.append(word_dict['**UNK**'])
            train_y_tuple.append(tag)
            if len(train_x_tuple) == 5:
                yield train_x_tuple, tag_dict[train_y_tuple[2]]
                # print("tagged sentence: {}.\ntrain_x:{}\ttag:{}".format(tagged_sentence, train_x_tuple, tag_dict[train_y_tuple[2]]))
                train_x_tuple = train_x_tuple[1:]
                train_y_tuple = train_y_tuple[1:]
                



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
    dev_file = path.join('..','pos','dev')
    with open(input_file,'rt') as i:
        word_dict, tag_dict = scan_train_for_vocab(i)
    word_dict['**START**'] = len(word_dict)
    word_dict['**STOP**'] = len(word_dict)
    word_dict['**UNK**'] = len(word_dict)
    tag_dict[''] = len(tag_dict)
    
    x, m, E, b, u, bp = build_ex1_network(len(word_dict), len(tag_dict))

    with open(dev_file, 'rt') as d:
        dev_set = list(generate_tagged_sentences(d))

    dev_set = dev_set[:len(dev_set)/10]

    #print(list(generate_train_tuples(dev_set[:10],word_dict, tag_dict)))
    print(tag_dict)
    with open(input_file,'rt') as i:
        train_data_iter = generate_train_tuples(generate_tagged_sentences(i), word_dict, tag_dict)

        # for j in range(5):
        #    train_data_iter.next()
        train_network(x, m, E, b, u, bp, train_data_iter, dev_set)
  
  
        # for tagged_word in sentence:
        #         train_network
        #     tagged_sentence_to_train_data
        #     codes_sentence = [(word_dict[a],tag_dict[b]) for a,b in sentence]
        #     print(sentence)
        #     print(codes_sentence)
            
        #     c += 1
        #     if c==5: 
        #         break
            