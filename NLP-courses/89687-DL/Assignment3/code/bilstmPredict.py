"""tag a test file with various types of trained networks
Usage:
    bilstmPredict.py <repr> <modelFile> <inputFile> 

Options:
    <repr>  the representation of the input, one of a,b,c,d
    <modelFile>  the base name of the file where the model is stored
    <inputFile>  the name of the input file. if the name resolves to a directory, the file name 'test' is assumed'
"""

import dynet as dy
from os import path
from docopt import docopt
from os import path
import bilstmTrain as bt
import pickle
import network_structure as networks

def read_untagged(fname, dummy_tag):
    sent = []
    for line in open(fname):
        line = line.strip()
        if not line:
            if sent: yield sent
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
    encoder_file = path.join(input_dir, bt.ENCODER_FILE)

    encoder = pickle.load(encoder_file)

    Repr = arguments["<repr>"]
    network_class = networks.choose_network_class(Repr)

    pc = dy.parameterCollection()
    params = iter(dy.load(input_file, pc))
    
    network =  network_class.load(pc,params, encoder)
    
    test_data = list(read_untagged(input_file,''))
    w = s = 0
    with open(output_file, 'rt') as a:
        for sent in tag_test_corpus(test_data, network, encoder):
            s += 1
            for word, tag in sent:
                w += 1
                a.write("{} {}\n".format(word,tag))
            a.write("\n") # space between sentences
    print("finished tagging {} word in {} sentences".format(w,s))
