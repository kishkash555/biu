"""Train various network types for tagging task
Usage:
    bilstmTrain.py <repr>

Options:
    <repr>  the representation of the input, one of a,b,c,d
"""



import numpy as np
import dynet as dy
from collections import Counter, OrderedDict
from numpy import random
from os import path
import datetime
import network_structure as networks
from docopt import docopt


"""Train various network types for tagging task
Usage:
    bilstmTrain.py <repr> <trainFile> <modelFile>

Options:
    <repr>  the representation of the input, one of a,b,c,d
    <trainFile>  the name of the input train data file
    <modelFile>  the name of the output model file
"""

UNK = '**UNK**'
INPUT_DIR = 'ner'

ACC_FILE = "best_accuracies.txt"
def read(fname):
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w,p = line
            sent.append((w,p))

def scan_train_for_vocab(train_data):
    words = Counter()
    tags = Counter()
    for line in train_data:
        if len(line) > 1:
            word, tag = line.strip().split()
            words[word] += 1
            tags[tag] += 1
        
    tag_dict = OrderedDict((a[0],i) for i, a in enumerate(tags.most_common()))
    word_dict = OrderedDict((a[0],i) for i, a in enumerate(words.most_common()))
    return word_dict, tag_dict


def load_and_start(network_class):
    input_path = path.abspath(path.join('..',INPUT_DIR))
    train_file = path.join(input_path,'train')
    dev_file = path.join(input_path, 'dev')

    train_data = list(read(train_file))
    dev_data = list(read(dev_file))
    with open(train_file, 'rt') as a:
        word_dict, tag_dict = scan_train_for_vocab(a)
    word_dict[UNK] = len(word_dict)
    print(tag_dict)
    # convert the whole set to numbers
    encoder = simpleCorpusEncoder(word_dict, tag_dict)
    train_codes = encoder.simpleEncode(train_data)
    print(train_codes[:3])
    dev_codes = encoder.simpleEncode(dev_data) 
    network = network_class(len(word_dict))
    train_network(train_codes, dev_codes, network)
    

class simpleCorpusEncoder:
    def __init__(self, word_dict, tag_dict):
        self.word_dict = word_dict
        self.tag_dict = tag_dict
    
    def simpleEncode(self, corpus):
        ret = [[(self.word_code(w), self.tag_dict[t]) for w, t in sent] for sent in corpus]
        return ret

    def word_code(self, word):
        if word in self.word_dict:
            return self.word_dict[word]
        return self.word_dict[UNK]



def single_training_pass(word_tag_pairs, network):
    tags_hat = network.evaluate_network(word_tag_pairs)
    errs = [dy.pickneglogsoftmax(t_h, t[1]) for t_h, t in zip(tags_hat, word_tag_pairs)]
    return dy.esum(errs)


def tag_sent(words, model, network):
    tags_hat = network.evaluate_network(words)
    chosen = [np.argmax(t_h.npvalue()) for t_h in tags_hat]
    return zip(words,chosen)

EPOCHS = 5
def train_network(train_data, dev_data, network):
    global prev_acc, prev_acc_ex0, run_id
    model = network.model
    prev_acc = prev_acc or 0.5
    prev_acc_ex0 = prev_acc_ex0 or 0.5
    trainer = dy.SimpleSGDTrainer(model)  
    tagged = loss = 0
    for ep in range(EPOCHS):
        random.shuffle(train_data)
        for i,s in enumerate(train_data,1):
            if i % 5000 == 0:
                # trainer.status()
                print("average loss last 5000 cycles: {}".format(loss / tagged))
                loss = 0
                tagged = 0
            if i>1000 and i % 1000 == 0:
                good_ex0 = good = bad = 0.0
                for sent in dev_data:
                    #print sent
                    tagged_sentence = tag_sent(sent, model, network)
                    tags = [t for w, t in sent]
                    tags_hat = [t for w, t in tagged_sentence]
                    for th, t in zip(tags_hat, tags):
                        if th == t: 
                            good +=1
                            if t > 0:
                                good_ex0 +=1  
                        else: 
                            bad += 1
                acc = good/(good+bad)
                acc_ex0 = good_ex0/(good_ex0+bad)
                print("dev accuracy after {} cycles: {}, {}".format(i, acc, acc_ex0))
                if acc > prev_acc:
                    #print("saving")
                    model.save("{}_{}.dy".format(INPUT_DIR,run_id)) 
                    prev_acc = acc
                if acc_ex0 > prev_acc_ex0:
                    prev_acc_ex0 = acc_ex0
            
            sum_errs = single_training_pass(s, network)
            loss += sum_errs.scalar_value()
            tagged += len(s)
            sum_errs.backward()
            trainer.update() 

if __name__ == "__main__":
    global prev_acc, prev_acc_ex0, run_id
    
    arguments = docopt(__doc__)
    
    if arguments["<repr>"] == "a":
        network_class = networks.part3aNetwork
    elif arguments["<repr>"] == "b":
        network_class = networks.part3bNetwork
    elif arguments["<repr>"] in {"c","d"}:
        raise NotImplementedError("please wait for the next epoch")
    else:
        raise ValueError("please specify a,b,c, or d")

    run_id = str(datetime.datetime.now().microsecond)
    try:
        a = open(ACC_FILE,"rt") 
        lines = a.readlines()
        acc_file_found = True
    except:
        print("{} not found".format(ACC_FILE))
        prev_acc = 0.5
    
    if acc_file_found:
        accs = [float(t.strip().split("\t")[3]) for t in lines]
        accs_ex0 = [float(t.strip().split("\t")[4]) for t in lines]

        prev_acc = max(accs)
        start_acc = prev_acc
        prev_acc_ex0 = max(accs_ex0)
        print("prev acc: {}".format(prev_acc))
        a.close()

    try:
        load_and_start(network_class)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if prev_acc > start_acc:
            with open(ACC_FILE,"at") as a:
                a.write("\t".join([
                    INPUT_DIR,
                    str(run_id),
                    datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S"),
                    "{:.3}".format(prev_acc),
                    "{:.3}".format(prev_acc_ex0)
                    ]))
                a.write("\n")
            