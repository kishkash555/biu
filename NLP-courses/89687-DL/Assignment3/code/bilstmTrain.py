
import numpy as np
import dynet as dy
from collections import Counter, OrderedDict
from numpy import random
from os import path
import datetime
UNK = '**UNK**'
INPUT_DIR = 'ner'

EMBEDDING_SIZE = 128
LSTM_HIDDEN_DIM = 50
LINEAR_DIM = 50
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


def load_and_start():
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
    model, params = init_network(len(word_dict))
    train_network(train_codes, dev_codes, model, params)
    
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

def init_network(nwords):
    model = dy.Model()
    params = {
        "E": model.add_lookup_parameters((nwords, EMBEDDING_SIZE)),
 #       "E_chars": model.add_lookup_parameters((nchars, EMBEDDING_SIZE)),
        "builders": [
            dy.LSTMBuilder(1, EMBEDDING_SIZE, LSTM_HIDDEN_DIM, model)
            for _ in range(2)
            ] + [
            dy.LSTMBuilder(1, LSTM_HIDDEN_DIM*2, LSTM_HIDDEN_DIM, model)
            for _ in range(2)    
            ],
        "W": model.add_parameters((LINEAR_DIM, LSTM_HIDDEN_DIM * 2 )),
        "v": model.add_parameters(LINEAR_DIM)
    }

    return model, params

def single_training_pass(word_tag_pairs, params):
    tags_hat = evaluate_network(word_tag_pairs, params)
    errs = [dy.pickneglogsoftmax(t_h, t[1]) for t_h, t in zip(tags_hat, word_tag_pairs)]
    return dy.esum(errs)

def evaluate_network(word_tag_pairs, params):
    dy.renew_cg()
    builders = params["builders"]
    E = params["E"]
    W = params["W"]
    v = params["v"]

    lstms = [b.initial_state() for b in builders]

    try:
        wembs = [E[w] for w, t in word_tag_pairs]
    except: 
        print(word_tag_pairs)
        raise
    # wembs = [dy.noise(we, 0.1) for we in wembs]

    # running the first level for getting b
    fw_lstm1 = lstms[0].transduce(wembs)
    bw_lstm1 = reversed(lstms[1].transduce(reversed(wembs)))

    inputs_to_2nd_layer = [dy.concatenate([f,b]) for f,b in zip(fw_lstm1,bw_lstm1)]
    
    fw_lstm2 = lstms[2].transduce(inputs_to_2nd_layer)
    bw_lstm2 = reversed(lstms[3].transduce(reversed(inputs_to_2nd_layer)))

    y = [dy.concatenate([f,b]) for f,b in zip(fw_lstm2,bw_lstm2)]
    tags_hat = [W * t + v for t in y]
    return tags_hat

def tag_sent(words, model, params):
    tags_hat = evaluate_network(words, params)
    chosen = [np.argmax(t_h.npvalue()) for t_h in tags_hat]
    return zip(words,chosen)

EPOCHS = 5
def train_network(train_data, dev_data, model, params):
    global prev_acc, run_id
    prev_acc = prev_acc or 0.5
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
                good = bad = 0.0
                for sent in dev_data:
                    #print sent
                    tagged_sentence = tag_sent(sent, model, params)
                    tags = [t for w, t in sent]
                    tags_hat = [t for w, t in tagged_sentence]
                    for th, t in zip(tags_hat, tags):
                        if th == t: 
                            good +=1 
                        else: 
                            bad += 1
                acc = good/(good+bad)
                print("dev accuracy after {} cycles: {}".format(i, acc))
                if acc > prev_acc:
                    #print("saving")
                    model.save("{}_{}.dy".format(INPUT_DIR,run_id)) 
                    prev_acc = acc
            
            
            sum_errs = single_training_pass(s, params)
            loss += sum_errs.scalar_value()
            tagged += len(s)
            sum_errs.backward()
            trainer.update() 

if __name__ == "__main__":
    global prev_acc, run_id
    run_id = str(datetime.datetime.now().microsecond)
    try:
        with open(ACC_FILE,"rt") as a:
            lines = a.readlines()
            accs = [float(t.strip().split("\t")[3]) for t in lines]
            prev_acc = max(accs)
            print("prev acc: {}".format(prev_acc))
    except:
        print("{} not found".format(ACC_FILE))
        prev_acc = 0.5
        raise

    try:
        load_and_start()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        with open(ACC_FILE,"at") as a:
            a.write("\t".join([
                INPUT_DIR,
                str(run_id),
                datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S"),
                "{:.3}".format(prev_acc)
                ]))
            a.write("\n")
            