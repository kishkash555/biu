"""tag a test file with various types of trained networks
Usage:
    bilstmPredict.py <repr> <modelFile> <inputFile> [--vocab=<vocabFile>] [--score_input]

Options:
    <repr>  the representation of the input, one of a,b,c,d
    <modelFile>  the base name of the file where the model is stored
    <inputFile>  the name of the input file. if the name resolves to a directory, the file name 'test' is assumed'
    --vocab=<vocabFile>  The name of the file where vocab is stored. If omitted, it is assumed to be in the same directory as the input file and named 'trainwords.pickle'
    --score_input  Treat input file as a dev file (with labels). instead of output, computer the classifier's score on it
"""

import dynet as dy
from os import path
from docopt import docopt
from os import path
import bilstmTrain as bt
import pickle
import network_structure as networks

corpusEncoder = bt.corpusEncoder

def read_untagged(fname, dummy_tag):
    sent = []
    with open(fname, 'rt') as a:
        for line in a:
            line = line.strip()
            if not line:
                if len(sent): yield sent
                sent = []
            else:
                w = line
                p = dummy_tag
            sent.append((w,p))

             
def tag_test_corpus(test_data, network, encoder):
    for sentence_to_tag in test_data:
        tagged_sent = bt.decode_tagged_sent(bt.tag_sent(sentence_to_tag, network), encoder)
        yield tagged_sent

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    input_file = path.abspath(arguments["<inputFile>"])
    
    if path.isdir(input_file):
        input_file = path.join(input_file,'test')

    print('input file: {}'.format(input_file))
    
    input_dir = path.dirname(input_file)
    output_file = path.join(input_dir, 'test_tagged')

    encoder_file = arguments["--vocab"] or path.join(input_dir, bt.ENCODER_FILE)

    with open(encoder_file, 'rb') as a:
        encoder = pickle.load(a)

    Repr = arguments["<repr>"]
    network_class = networks.choose_network_class(Repr)

    model_file = arguments["<modelFile>"]
    pc = dy.ParameterCollection()
    params = iter(dy.load(model_file, pc))
    
    network =  network_class.load(pc,params, encoder)
    
    dummy_tag = encoder.tag_dict.keys()[-1]
    #print("test data", test_data[:20])
    w = s = 0
    if arguments["--score_input"]:
        acc, acc_ex0 = bt.test_a_classifier_on_dev(network,bt.read(input_file))
        print("accuracy on dev: {}, excluding most common: {}".format(acc,acc_ex0))
    else:
        test_data = list(read_untagged(input_file,dummy_tag))
        with open(output_file, 'wt') as a:
            for sent in tag_test_corpus(test_data, network, encoder):
                s += 1
                for word, tag in sent:
                    w += 1
                    a.write("{} {}\n".format(word[0],tag))
                a.write("\n") # space between sentences
        print("finished tagging {} word in {} sentences".format(w,s))