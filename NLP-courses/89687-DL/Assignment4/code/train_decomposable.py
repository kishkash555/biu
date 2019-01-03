#!/usr/bin/python
import decomposable as dec
import parsers
import sys

GLOVE_FILE = '../glove_filtered00.txt'
SNLI_TRAIN = '../snli_1.0/snli_1.0_train_stripped.txt'
SNLI_DEV = '../snli_1.0/snli_1.0_dev_stripped.txt'

import _dynet as dy

dyparams = dy.DynetParams()
dyparams.set_autobatch(True)
# dyparams.set_mem(4096)
dyparams.init()


if __name__ == "__main__":
    with open(GLOVE_FILE, 'rt') as a:
        glove = parsers.glove_embeddings(a)

    max_cases = 0
    try:
        max_cases = int(sys.argv[1])
    except:
        pass

    train_data, labels = parsers.load_snli(SNLI_TRAIN, max_cases)
    dev_data, labels = parsers.load_snli(SNLI_DEV, labels=labels)
    print labels
    print dev_data[10]
    print "** done loading"
    my_net = dec.decomposable(glove)
    print "** done init"
    my_net.train_network(train_data,5, dev_data)
