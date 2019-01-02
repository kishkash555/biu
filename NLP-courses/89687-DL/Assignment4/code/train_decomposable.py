#!/usr/bin/python
import decomposable as dec
import parsers
import dynet
GLOVE_FILE = '../glove_filtered_dev.txt'
SNLI_TRAIN = '../snli_1.0/snli_1.0_train_stripped.txt'
SNLI_DEV = '../snli_1.0/snli_1.0_dev_stripped.txt'

if __name__ == "__main__":
    with open(GLOVE_FILE, 'rt') as a:
        glove = parsers.glove_embeddings(a)

    train_data = parsers.load_snli(SNLI_TRAIN)
    dev_data = parsers.load_snli(SNLI_DEV)
    print "** done loading"
    my_net = dec.decomposable(glove)
    print "** done init"
    my_net.train_network(train_data)
