
WORD_EMBEDDING_SIZE = 128
CHAR_EMBEDDING_SIZE = 40
LSTM_HIDDEN_DIM = 50
LINEAR_DIM = 60

import _dynet as dy

class part3aNetwork:
    def __init__(self, nwords=None, encoder=None):
        if not nwords and not encoder:
            return
        self.common = commonNetwork(WORD_EMBEDDING_SIZE)
        self.model = self.common.model
        self.encoder = encoder
        if not nwords:
            nwords = len(encoder.word_dict)
        self.params = {
            "E": self.model.add_lookup_parameters((nwords, WORD_EMBEDDING_SIZE)),
        }
    
    def params_iterable(self):
        common_params = self.common.params_iterable()
        for param in common_params:
            yield param
        yield self.params["E"]
    
    def save(self, basefile):
        dy.save(basefile, self.params_iterable())

        
    @classmethod
    def load(cls, model, params, encoder):
        self = cls()
        self.common = commonNetwork.load(model,params)
        self.params = {
            "E": next(params)
        }
        self.encoder = encoder
        return self
   
    def evaluate_network_from_codes(self, word_tag_pairs):
        dy.renew_cg()
        E = self.params["E"]
        input_vectors = [E[w] for w, t in word_tag_pairs]
        return self.common.evaluate_network_from_embs(input_vectors, False)

    def evaluate_network_from_sentence(self, sentence):
        coded_sent = self.encode_sentence(sentence)
        #print("sentence: {}\ncoded: {}".format(sentence, coded_sent))
        return self.evaluate_network_from_codes(coded_sent)

    def encode_sentence(self, sentence):
        return self.encoder.encode_sentence_words(sentence)


class commonNetwork:
    def __init__(self, input_vector_size, *argc):
        if input_vector_size == 0:
            return
        model = dy.Model()
        self.params = {
            "builders": [
                dy.LSTMBuilder(1, input_vector_size, LSTM_HIDDEN_DIM, model)
                for _ in range(2)
                ] + [
                dy.LSTMBuilder(1, LSTM_HIDDEN_DIM*2, LSTM_HIDDEN_DIM, model)
                for _ in range(2)    
                ],
            "W": model.add_parameters((LINEAR_DIM, LSTM_HIDDEN_DIM * 2 )),
            "v": model.add_parameters(LINEAR_DIM)
        }
        self.model = model

    def params_iterable(self):
        for i in range(4):
            yield self.params["builders"][i]
        yield self.params["W"] 
        yield self.params["v"]
    
    @classmethod
    def load(cls, model, params):
        self = cls(0)
        self.model = model
        self.params = {
            "builders": [next(params) for _ in range(4)],
            "W": next(params),
            "v": next(params)
        }
        return self
    
   

    def evaluate_network_from_embs(self, wembs, renew=True):
        params = self.params
        if renew:
            dy.renew_cg()
        builders = params["builders"]
        W = params["W"]
        v = params["v"]

        lstms = [b.initial_state() for b in builders]

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



class part3bNetwork:
    def __init__(self, nchars=None, encoder=None):
        if not nchars and not encoder:
            return
        self.common = commonNetwork(LSTM_HIDDEN_DIM)
        self.model = self.common.model
        self.encoder = encoder
        if not nchars:
            nchars = len(encoder.char_dict)
        self.params = {
            "builders": [
                dy.LSTMBuilder(1, CHAR_EMBEDDING_SIZE, LSTM_HIDDEN_DIM, self.model)
                ],
            "E": self.model.add_lookup_parameters((nchars, CHAR_EMBEDDING_SIZE))
        }

    def params_iterable(self):
        common_params = self.common.params_iterable()
        for param in common_params:
            yield param
        yield self.params["builders"][0]
        yield self.params["E"]
        
    @classmethod
    def load(cls, model, params, encoder):
        self = cls()
        self.common = commonNetwork.load(model,params)
        self.params = {
            "builders": [next(params)],
            "E": next(params)
        }
        self.encoder = encoder
        return self
   
    def save(self, basefile):
        dy.save(basefile, self.params_iterable())

    def evaluate_network_from_codes(self, sentence_as_char_codes):
        params = self.params
        dy.renew_cg()
        builder = params["builders"][0]
        E = params["E"]
        sentence_as_wembs = []
        for word in sentence_as_char_codes:
            char_lstm = builder.initial_state()
            cembs = [E[char] for char in word]

        # running the char-level lstm
            word_vec = char_lstm.transduce(cembs)[-1]
            sentence_as_wembs.append(word_vec)
        
        return self.common.evaluate_network_from_embs(sentence_as_wembs,False)
    
    def evaluate_network_from_sentence(self, sentence):
        coded_sent = self.encode_sentence(sentence)
        #print("sentence: {}\ncoded: {}".format(sentence, coded_sent))
        return self.evaluate_network_from_codes(coded_sent)


    def encode_sentence(self,sentence):
        """
        get sentence as list of words (strings)
        return sentence as list of list of char encodings
        """
        return self.encoder.encode_sentence_chars(sentence)

def choose_network_class(Repr):
    if Repr == "a":
        return part3aNetwork
    elif Repr == "b":
        return part3bNetwork
    elif Repr in {"c","d"}:
        raise NotImplementedError("please wait for the next epoch")
    else:
        raise ValueError("please specify a,b,c, or d")
