import dynet as dy
import numpy as np
from collections import OrderedDict
vocab = list("123456789abcd") + ["<EOS>"]
VOCAB_SIZE = len(vocab)

char2int = OrderedDict(zip(vocab,range(VOCAB_SIZE)))

NUM_LAYERS=1
INPUT_DIM=50
HIDDEN_DIM=10

# not in use
def train_lstm(train, lstm_builder):
    """
    evaluate lstm
    """
    x = dy.vecInput(INPUT_DIM)
    s_previous = lstm_builder.initial_state()
    for train_x, train_y in train:
        for char in train_x:
            x.set(one_hot(vocab_dict[train_x],INPUT_DIM))
            s_next = s_previous.add_input
            s_previous = s_next
        y = s_next.output().npvalue()


def do_one_sentence(lstm, params, sentence):
    """
    this code was copied from https://dynet.readthedocs.io/en/latest/tutorials_notebooks/RNNs.html#Character-level-LSTM and slightly modified
    """
    # setup the sentence
    dy.renew_cg()
    s0 = lstm.initial_state()


    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]
    sentence = ["<EOS>"] + list(sentence) + ["<EOS>"]
    sentence = [char2int[c] for c in sentence]
    s = s0
    loss = []
    for char,next_char in zip(sentence,sentence[1:]):
        s = s.add_input(lookup[char])
        probs = dy.softmax(R*s.output() + bias)
        loss.append( -dy.log(dy.pick(probs,next_char)) )
    loss = dy.esum(loss)
    return loss


# train, and generate every 5 samples
def train(lstm, params, sentence):
    trainer = dy.SimpleSGDTrainer(pc)
    for i in range(400):
        loss = do_one_sentence(lstm, params, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 10 == 0:
            print("%.10f" % loss_value + "\t" + generate(lstm, params))


# generate from model:
def generate(rnn, params):
    def sample(probs):
        rnd = np.random.random()
        for i,p in enumerate(probs):
            rnd -= p
            if rnd <= 0: break
        return i

    # setup the sentence
    dy.renew_cg()
    s0 = rnn.initial_state()

    R = params["R"]
    bias = params["bias"]
    lookup = params["lookup"]
    int2char = list(char2int.keys())

    s = s0.add_input(lookup[char2int["<EOS>"]])
    out=[]
    while True:
        probs = dy.softmax(R*s.output() + bias)
        probs = probs.vec_value()
        next_char = sample(probs)
        out.append(int2char[next_char])
        if out[-1] == "<EOS>": break
        s = s.add_input(lookup[next_char])
    return "".join(out[:-1]) # strip the <EOS>


def add_params():
    # add parameters for the hidden->output part for both lstm and srnn
    params = {}
    #for params in [params_lstm, params_srnn]:
    params["lookup"] = pc.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
    params["R"] = pc.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
    params["bias"] = pc.add_parameters((VOCAB_SIZE))
    return params


def one_hot(k,m):
    ret = np.zeros(m)
    ret[k]=1
    return ret


def lstm():
    pc = dy.ParameterCollection()
    builder = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
    return builder

if __name__ == "__main__":
    pc = dy.ParameterCollection()
    lstm = dy.LSTMBuilder(NUM_LAYERS, INPUT_DIM, HIDDEN_DIM, pc)
    #sentence = "43435311277561741167761273524725475142356832367aaaaaaaaaaaaaaaaaaa48541816466562542351725463718ccccccccccccccccccc5bbbbbbbbbbbbbbbbb4886263441411373dddddddddddd"
    sentence = "aaaaabbbbbcccccddddd1234567"
    params_lstm = add_params()
    
    train(lstm, params_lstm, sentence)