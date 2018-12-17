"""Train various network types for tagging task
Usage:
    bilstmTrain.py <repr> <trainFile> <modelFile> [--dev=<devfile>]

Options:
    <repr>  the representation of the input, one of a,b,c,d
    <trainFile>  the name of the training file. If the name resolves to a directory, the file name 'train' is assumed.
    <modelFile>  the name of the file to save the model.
    --dev=<devfile>  the name of the dev file (for performance evaluation during training). assumed in same dir as train file.
    --dir=<dirname>  the directory of the train file (named 'train'), dev file (named 'dev') and output model file (named 'trained_model') 
"""



import numpy as np
import dynet as dy
from collections import Counter, OrderedDict
from numpy import random
from os import path
import datetime
import network_structure as networks
from docopt import docopt
import json
import time
import pickle

"""Train various network types for tagging task
Usage:
    bilstmTrain.py <repr> <trainFile> <modelFile>

Options:
    <repr>  the representation of the input, one of a,b,c,d
    <trainFile>  the name of the input train data file
    <modelFile>  the name of the output model file
"""

UNK = '**UNK**'
#INPUT_DIR = 'pos'
ENCODER_FILE = 'trainwords.pickle'
ACC_FILE = "best_accuracies.txt"
REPORT_FILE =  "report.json"
PREFIX_LENGTH = 3
SUFFIX_LENGTH = 3

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


def load_and_start(network_class, train_file, dev_file):
    global task_id
    train_data = list(read(train_file))
    dev_data = list(read(dev_file))
    print(len(train_data))
    print(len(dev_data))
    with open(train_file, 'rt') as a:
        word_dict, tag_dict = scan_train_for_vocab(a)
    word_dict[UNK] = len(word_dict)
    print(tag_dict)
    # convert the whole set to numbers
    encoder = corpusEncoder(word_dict, tag_dict)
    network = network_class(None, encoder)

    with open(task_id + "_" + ENCODER_FILE,'wb') as a:
        pickle.dump(encoder, a)
    print("saved encoder")
    train_network(train_data, dev_data, encoder, network)
    

class corpusEncoder:
    def __init__(self, word_dict, tag_dict):
        self.word_dict = word_dict
        self.tag_dict = tag_dict

        corpus_charset = set(''.join(word_dict.keys()))
        self.char_dict = OrderedDict([(c, i) for i,c in enumerate(corpus_charset)])
        self.char_dict[UNK] = len(self.char_dict)
        
        self.unk_words = set(word_dict.keys()[-1000:])
        self.unk_replace_prob = 0.2
    
        corpus_prefix_set = set([w[:PREFIX_LENGTH] for w in word_dict.keys() if len(w)-PREFIX_LENGTH>=2])
        corpus_suffix_set = set([w[-SUFFIX_LENGTH] for w in word_dict.keys() if len(w)-SUFFIX_LENGTH>=2])

        self.prefix_dict = OrderedDict([(p, i) for i,p in enumerate(corpus_prefix_set)])
        self.suffix_dict = OrderedDict([(s, i) for i,s in enumerate(corpus_suffix_set)])

        self.pre_len, self.suf_len = PREFIX_LENGTH, SUFFIX_LENGTH

    def encode_corpus_words(self, corpus):
        ret = [self.encode_sentence_words(sent) for sent in corpus]
        return ret

    def encode_sentence_words(self, sentence):
        ret = [(self.word_code(w), self.tag_dict[t]) for w, t in sentence]
        return ret

    def encode_sentence_words_with_unks(self, sentence):
        ret = [(self.word_code_with_unks(w), self.tag_dict[t]) for w, t in sentence]
        return ret

    def encode_sentence_chars(self, sentence):
        ret = [[self.char_dict.get(c,self.char_dict[UNK]) for c in w] for w, _ in sentence]
        return ret
    
    def encode_sentence_prefix_suffix(self, sentence):
        pre = lambda w: w[:self.pre_len]
        suf = lambda w: w[-self.suf_len:]
        ret = [
            (self.prefix_dict.get(pre(w), -1), self.suffix_dict.get(suf, -1))
         for w, _ in sentence
         ]
        return ret

    def word_code(self, word):
        return self.word_dict.get(word,self.word_dict[UNK])

    def word_code_with_unks(self, word):
        if word in self.unk_words and np.random.random() < self.unk_replace_prob:
            return self.word_dict[UNK]
        return self.word_code(word)


def single_training_pass(sentence, encoder, network):
    tags_hat = network.evaluate_network_from_sentence(sentence)
    word_tag_codes = encoder.encode_sentence_words_with_unks(sentence)
    errs = [dy.pickneglogsoftmax(t_h, t[1]) for t_h, t in zip(tags_hat, word_tag_codes)]
    #errs = [err for err, t in zip(errs,word_tag_codes) if t[1] != 'O' or np.random.random() < 0.1]
    return dy.esum(errs)


def tag_sent(words, network):
    tags_hat = network.evaluate_network_from_sentence(words)
    chosen = [np.argmax(t_h.npvalue()) for t_h in tags_hat]
    return zip(words,chosen)
    
def decode_tagged_sent(sent, encoder):
    tags_decoded = [encoder.tag_dict.keys()[t] for _, t in sent]
    return [(s[0],t) for s, t in zip(sent, tags_decoded)]

EPOCHS = 5
def train_network(train_data, dev_data, encoder, network):
    global prev_acc, prev_acc_ex0, model_file, report
    model = network.model
    trainer = dy.SimpleSGDTrainer(model)  

    prev_acc = prev_acc or 0.5
    prev_acc_ex0 = prev_acc_ex0 or 0.5
    
    report = []
    tagged = loss = 0
    i = 1
    t0 = time.clock()
    for ep in range(EPOCHS):
        random.shuffle(train_data)
        for s in train_data:
            i += 1
            if i % 500 == 0:
                print("average loss last 500 cycles: {}".format(loss / tagged))
                good_ex0 = good = bad = 0.0
                for sent in dev_data:
                    #print sent
                    tagged_sentence = tag_sent(sent, network)
                    # print(tagged_sentence)
                    tags = [t for w, t in encoder.encode_sentence_words(sent)]
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
                ti = time.clock() 
                report.append(OrderedDict([
                    ("cycles", i),
                    ("dev_accuracy", acc),
                    ("dev_accuracy_except_common", acc_ex0),
                    ("loss", loss / tagged),
                    ("clock_time",  round(ti-t0,2)),
                    ("saved", 0)
                ]))
                loss = 0
                tagged = 0
                ti = t0
                if acc > prev_acc:
                    print("saving")
                    network.save(model_file)
                    report[-1]["saved"] = 1
                    prev_acc = acc
                if acc_ex0 > prev_acc_ex0:
                    prev_acc_ex0 = acc_ex0
            
            sum_errs = single_training_pass(s, encoder, network)
            loss += sum_errs.scalar_value()
            tagged += len(s)
            sum_errs.backward()
            trainer.update() 
    


if __name__ == "__main__":
    global prev_acc, prev_acc_ex0, run_id, task_id, model_file, report

    arguments = docopt(__doc__)
    print(arguments)
    Repr = arguments["<repr>"]
    network_class = networks.choose_network_class(Repr)

    run_id = str(datetime.datetime.now().microsecond)
    acc_file_found = False
    try:
        a = open(ACC_FILE,"rt") 
        lines = a.readlines()
        acc_file_found = True
    except:
        print("{} not found".format(ACC_FILE))
        prev_acc = 0.5
        prev_acc_ex0 = 0.5
    train_file = arguments["<trainFile>"]
    if path.isdir(train_file):
        train_file = path.join(train_file, 'train')
    input_path = path.dirname(train_file)
    model_file = arguments["<modelFile>"]
    dev_basename = path.basename(arguments.get("--dev") or '') or 'dev' 
    dev_path = path.dirname(arguments.get("--dev") or train_file)
    dev_file = path.join(dev_path, dev_basename)
    task_id = path.split(input_path)[-1]

    if acc_file_found:
        
        lines = [line.strip().split("\t") for line in lines]
        lines = [line for line in lines if line[0]==task_id and line[5]==Repr ]
        accs = [float(t[3]) for t in lines ]
        accs_ex0 = [float(t[4]) for t in lines ]

        prev_acc = max(accs+[0.5])
        start_acc = prev_acc
        prev_acc_ex0 = max(accs_ex0+[0.5])
        print("prev acc: {}".format(prev_acc))
        a.close()
  
    
    load_and_start(network_class, train_file, dev_file)
    
    try:
        pass
    except KeyboardInterrupt:
        print("\nInterrupted")
    except:
        raise
    finally:
        if acc_file_found:
            report = [{
                "run_id": run_id, 
                "input_dir": task_id,
                "when":  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "best_acc": "{:.3}".format(prev_acc),
                "best_acc_ex0": "{:.3}".format(prev_acc_ex0),
                "network type": Repr,
                } ] + report
            with open(REPORT_FILE,"at") as a:
                json.dump(report,a)
                a.write('\n******\n')
            if prev_acc > start_acc:
                with open(ACC_FILE,"at") as a:
                    a.write("\t".join([
                        task_id,
                        str(run_id),
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "{:.3}".format(prev_acc),
                        "{:.3}".format(prev_acc_ex0),
                        Repr,
                        ]))
                    a.write("\n")
                