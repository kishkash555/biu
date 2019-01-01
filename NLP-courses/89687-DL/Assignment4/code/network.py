import _dynet as dy
import numpy as np

class glove_embeddings:
    def __init__(self, glove_lines):
        self.embeddings = self.parse_glove_lines(glove_lines)
        self.vec_dim = next(dictionary.itervalues()).shape[0]
        self.unk_vec = self.embeddings['<unk>']
        self.word_to_ind = {i: word for i, word in enumerate(self.embeddings.keys())}

    @classmethod
    def parse_glove_lines(cls, glove_lines):
        ret = {}
        for line in lines:
            word, data = line.strip().split(' ',1)
            ret[word]= np.fromstring(data, sep=' ') 
        return ret
    
    def get(self, word):
        self.embeddings.get(word, self.unk_vec)
    
    def as_numpy_array(self):
        return np.array(self.values())

    def as_dynet_lookup(self,pc):
        pc.add_lookup_parameters((len(self.embeddings,self.vec_dim)),  init = self.as_numpy_array())


class network:
    @classmethod
    def load(self, fname):
        raise NotImplementedError()

    def evaluate(self, *args):
        raise NotImplementedError()

    def train_network(self, train_data, epochs = 3):
        trainer = dy.AdamTrainer(self.pc)
        i = 0
        mloss = 0.
        goods = 0.
            
        for e in range(epochs):
            shuffle(train_data)
            for x, y in train_data:
                i = i + 1
                loss = self.eval_loss(x, y)
                good = y == self.last_case_class()
                mloss += loss.value()
                goods += int(good)
                loss.backward()
                trainer.update()
        print("average loss: {} acc: {}".format(mloss/i, goods/i))


class mlp_subnetwork():
    def __init__(self, pc, layer_sizes, hidden_activation, output_activation, is_head):
        self.pc = pc
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)
        self.input_size = layer_sizes[0]
        self.hidden_activation = hidden_activation
        self. output_activation = output_activation
        self.is_head = is_head
        params = [(
                pc.add_parameters((a,b), name = "W{:02d}".format(i)), 
                pc.add_parameters(a, name = "b{:02d}".format(i))
                ) for (i, a, b) in zip(range(len(layer_sizes)-1, layer_sizes[1:], layer_sizes[:-1]))
                    ]
        self.params = { 
            "W": [p[0] for p in params],
            "b": [p[1] for p in params]
        }


    def check_input_size(self, x_list):
        for x_np in x_list:
            x_size = x_np.squeeze().shape[0]
            if  x_size != self.input_size:
                raise ValueError("Expected input of size {} got {}".format(self.input_size, x_size))
        

    def evaluate_network(self, x_np, apply_final_activation=True):
        """
        return an expression that is the result of feeding the input through the entire 
        network, except the last activation
        """
        self.check_input_size(x_np)

        if self.is_head:
            dy.renew_cg()

        if type(x_np) == np.ndarray:
            x = dy.vecInput()
            x.set(x_np)
        elif type(x_np) == list:
            x = dy.inputTensor(x_list, batched = True)

        final_activation = self.output_activation if apply_final_activation else lambda x: x
        activation = self.hidden_activation
        for i, W, b in zip(range(n_layers), self.params["W"], self.params["b"]):
            if i == n_layers-1:
                    activation = final_activation
            x = activation(W*x + b)
        return x
    

    
    def eval_loss(self, x, y):
       # y_hat = self.evaluate_network(x)
        raise NotImplementedError()



