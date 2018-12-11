import dynet as dy
import numpy as np
import random
from collections import OrderedDict

use_EOS = False

vocab = list("123456789abcd")
if use_EOS:
    vocab += ["<EOS>"]

VOCAB_SIZE = len(vocab)

char2int = OrderedDict(zip(vocab,range(VOCAB_SIZE)))

NUM_LAYERS = 1
INPUT_DIM = 10
LSTM_HIDDEN_DIM = 10
MLP_HIDDEN_DIM = 15

# z = Softmax(U*tanh(W*y+b1)+b0)
get_param_dim = { 
    "lookup": (VOCAB_SIZE, INPUT_DIM),
    "W": (MLP_HIDDEN_DIM, LSTM_HIDDEN_DIM),
    "U": (2, MLP_HIDDEN_DIM),
    "b1": MLP_HIDDEN_DIM,
    "b0": 2
    }  


def do_one_sentence(lstm, params, sentence, train_y):
    """
    this code was copied from https://dynet.readthedocs.io/en/latest/tutorials_notebooks/RNNs.html#Character-level-LSTM and slightly modified
    """
    # before starting any dy computation
    dy.renew_cg()
    
    # set up the lstm
    s = lstm.initial_state()

    # network paramaters as variables
    ## lstm parameters
    lookup = params["lookup"]
    
    ## MLP parameters
    W = params["W"]
    b0 = params["b0"]
    U = params["U"]
    b1 = params["b1"]

    # print(sentence)
    sentence =  list(sentence) 
    if use_EOS:
        sentence = ["<EOS>"] + sentence + ["<EOS>"]

    # the code line below performs two tasks:
    # 1. maps the string (character) literals to their numeric codes
    # 2. maps the numeric codes to their embedding vectors
    xs = [lookup[char2int[c]] for c in sentence]
    
    # run the entire lstm (feeding one embedding vector at each stage). 
    # only the final output is needed
    y = s.transduce(xs)[-1]
    # print(train_y)
    output = dy.softmax(U * (dy.tanh(W*y + b1)) + b0)
    #output =y # !!!!!
    #print("train_y: {}. output: {}".format(train_y, dy.pick(output.npvalue(),train_y)))
    loss = -dy.log(dy.pick(dy.softmax(output),train_y))
    # if np.isnan(loss.value()):
    #     print("nan loss")
    # else:
        # print("-")
    y_hat = output.npvalue()
    return loss, y_hat


# train, and generate every 5 samples
def train(lstm, params, train_data, dev_data, epochs):
    trainer = dy.SimpleSGDTrainer(pc)
    for _ in range(epochs):
        np.random.shuffle(train_data)
        #print("train_data {}".format(train_data[0]))
        i = 0
        for train_y, sentence in train_data:
            #print("sentence\n{}\ntrain_y{}".format(sentence, train_y))
            loss, _ = do_one_sentence(lstm, params, sentence, train_y)
            #print("after do one sent")
            loss.backward()
            trainer.update()
            if i % 20 == 0:
               
                _, y_hat = do_one_sentence(lstm, params, sentence, train_y)
                #print("sentence: {} y_hat: {} y: {}".format(sentence, y_hat, train_y))
                dev_y, dev_sent = random.choice(dev_data)
                _, y_hat = do_one_sentence(lstm, params, dev_sent, dev_y)
                #print("dev sentence: {} y_hat: {} y: {}".format(dev_sent, y_hat, dev_y))
                dev_loss, dev_acc = check_loss(lstm, params, dev_data)
                print("%.10f" % dev_loss + "\t" + "%.2f" % dev_acc)
                emb = params["lookup"].npvalue()
                emb_col_norms = np.sqrt((emb * emb).sum(axis=0))
                emb = emb / emb_col_norms[np.newaxis,:] 
                emb_a = emb[:,9]
                emb_1 = emb[:,0]
                #print(emb.T.dot(emb_a))
                #print(emb.T.dot(emb_1))
            i += 1

def check_loss(lstm, params, dev_data):
    loss = 0 
    good = 0
    for dev_y, sentence in dev_data:
        curr_loss, y_hat = do_one_sentence(lstm, params, sentence, dev_y)
        y_hat = np.argmax(y_hat)
        loss += curr_loss.value()
        if y_hat == dev_y:
            good += 1
    return loss/len(dev_data), float(good)/len(dev_data) 


def add_params(pc):
    """
    construct MLP(LSTM(E(x)))
    E - an embedding matrix trained together with the entire network
    The MLP expression is:
    z = Softmax(W*tanh(U*y+b1)+b0)
    where b1, b0 are bias vectors
    U is the hidden layer matrix
    W is the output layer matrix
    y is the MLP input which is the lstm output
    """
    # add parameters for the hidden->output part for both lstm and srnn
    params = {}
    # LSTM params
    params["lookup"] = pc.add_lookup_parameters(get_param_dim["lookup"])

    # MLP params
    params["W"] = pc.add_parameters(get_param_dim["W"])
    params["U"] = pc.add_parameters(get_param_dim["U"])
    params["b0"] = pc.add_parameters(get_param_dim["b0"])
    params["b1"] = pc.add_parameters(get_param_dim["b1"])
    return params


def lstm():
    pc = dy.ParameterCollection()
    dy.AdamTrain(pc)
    builder = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, LSTM_HIDDEN_DIM, pc)
    return builder

def load_train(fname):
    train_data = []
    with open(fname,"rt") as a:
        for line in a:
            y, x = line.strip().split()
            train_data.append((int(y),x)) 
    return train_data

if __name__ == "__main__":
    pc = dy.ParameterCollection()
    lstm = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, LSTM_HIDDEN_DIM, pc)
    train_data = load_train("train1")
    #print(train_data[:5])
    dev_data = load_train("dev1")
    params = add_params(pc)
    train(lstm, params, train_data, dev_data, 3)
