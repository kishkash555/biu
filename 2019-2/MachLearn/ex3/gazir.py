import numpy as np
from scipy.special import softmax
from collections import OrderedDict, namedtuple

train_options_nt = namedtuple('train_options',['epochs','report_interval'])

class layer:
    def __init__(self):
        self.ctx = {}

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad_output):
        raise NotImplementedError()
    
    def grad_loss(self, y):
        raise NotImplementedError()

    def update(self, stepsize):
        pass # creates a default of not doing anything


class linear_layer(layer):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.parameters = OrderedDict([
            ('W', np.random.rand(fan_in, fan_out)),
            ('b', np.random.rand(fan_out))
        ])
    
    def forward(self, x):
        W, b = self.parameters.values()
        self.ctx['input'] = x.copy()
        return x.dot(W) + b

    def backward(self, grad_output):
        W, _ = self.parameters.values()
        x = self.ctx['input']
        grad_input = grad_output.dot(W.T)
        grad_weight = x.T.dot(grad_output)
        grad_bias = np.sum(grad_output,axis=0)
        self.ctx['grads'] = grad_weight, grad_bias
        return grad_input

    def update(self, stepsize):
        W, b = self.parameters.values()
        grad_weight, grad_bias = self.ctx['grads']
        W += grad_weight
        b += grad_bias
    

class relu_layer(layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        self.ctx['input'] = x.copy()
        x[x<0.] =0.
        return x

    def backward(self, grad_output):
        ret = grad_output.copy()
        x = self.ctx['input']
        ret[x<0.] = 0.
        return ret


class softmax_nll_layer(layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ret = softmax(x, axis=1)
        self.ctx['softmax'] = ret.copy()
        return ret

    def backward(self, grad_output): 
        pass
    
    def grad_loss(self, y):
        ret = self.ctx['softmax']
        ret[:,y] = ret[:,y] - 1
        # ret = np.sum(ret,axis=0)
        return ret



class network(layer):
    def __init__(self, layer_obj, layer_names = None):
        super().__init__()
        layer_names = layer_names or ['layer{:02}'.format(i) for i in range(len(layer_obj))]
        self.layers = OrderedDict(zip(layer_names, layer_obj))
        self.train_options = train_options_nt(20, 2000)
    
    def set_train_options(self, **kwargs):
        self.train_options = self.train_options._replace(**kwargs)

    def forward(self, x):
        z = x
        for layer_obj in self.layers.values():
            z = layer_obj.forward(z)
        return z

    def backward(self, y):
        is_last_layer = True
        for layer_obj in reversed(self.layers.values()):
            if is_last_layer:
                grad = layer_obj.grad_loss(y)
                is_last_layer = False
            else:
                grad = layer_obj.backward(grad)
        return grad
        
    def update(self, stepsize):
        for layer_obj in self.layers.values():
            layer_obj.update(stepsize)

    def train(self, train_set, lr, validation_set = None):
        """
        train_set: an iterable of x,y pairs, possibly batched
        lr: learning rate object 
        validation_set: an iterable of x,y pairs (possibly batched) for checking the accuracy
        """
        lr_gen = lr.lr_generator
        to = self.train_options
        for ep in range(to.epochs):
            for i, (x, y) in enumerate(iter(train_set)):
                st = next(lr_gen)
                self.forward(x)
                self.backward(y)
                self.update(st)
                if i % to.report_interval == 0:
                    # report code here
                    pass 
            lr.new_epoch()

