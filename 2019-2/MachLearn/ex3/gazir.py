import numpy as np
from scipy.special import softmax
from collections import OrderedDict, namedtuple
import time
import pickle

train_options_nt = namedtuple('train_options',['epochs','report_interval'])

def time2str(st):
    return time.strftime('%H:%M:%S',time.localtime(st))

class layer:
    def __init__(self):
        self.ctx = {}

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad_output):
        raise NotImplementedError()

    def update(self, stepsize):
        pass # creates a default of not doing anything

class loss_layer(layer):
    def __init__(self):
        super().__init__()

    def grad_loss(self, y):
        raise NotImplementedError()
    
    def get_loss(self):
        raise NotImplementedError


class linear_layer(layer):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.parameters = OrderedDict([
            ('W', np.random.rand(fan_in, fan_out)*2-1),
            ('b', np.random.rand(fan_out)*2-1)
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
        W -= grad_weight * stepsize
        b -= grad_bias * stepsize
    

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


class dropout_layer(layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate
    
    def forward(self, x):
        self.ctx['mask'] = np.random.rand(x.shape[1]) > self.rate
        self.ctx['factor'] = x.shape[1] / self.ctx['mask'].sum()
        ret = x.copy()
        ret[:,np.logical_not(self.ctx['mask'])] = 0.
        ret *= self.ctx['factor']
        return ret
 
    def backward(self, grad_output):
        ret = grad_output.copy()
        ret[:,np.logical_not(self.ctx['mask'])] = 0.
        ret *= self.ctx['factor']
        return ret

class softmax_nll_layer(loss_layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ret = softmax(x, axis=1)
        self.ctx['softmax'] = ret.copy()
        return ret

    def backward(self, grad_output): 
        pass
    
    def grad_loss(self, y):
        ret = self.ctx['softmax'].copy()
        rows = range(ret.shape[0])
        self.ctx['loss'] = -np.log(ret[rows,y])
        if np.any(self.ctx['loss'] == np.inf):
            print('inf detected. softmax:\n{}\nloss:{}'.format(ret[rows,y], self.ctx['loss']))
        ret[rows,y] = ret[rows,y] - 1
        return ret

    def get_loss(self):
        return self.ctx['loss'].sum()

class network(loss_layer):
    def __init__(self, layer_obj, layer_names = None):
        super().__init__()
        layer_names = layer_names or ['layer{:02}'.format(i) for i in range(len(layer_obj))]
        self.layers = OrderedDict(zip(layer_names, layer_obj))
        self.train_options = {"epochs": 20, "report_interval": 50} 
    
    def set_train_options(self, **kwargs):
        for k, v in kwargs.items():
            self.train_options[k] = v

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

    def grad_loss(self,y):
        return next(reversed(self.layers.values())).grad_loss(y)
    
    def get_loss(self):
        return next(reversed(self.layers.values())).get_loss()
        
    def train(self, train_set, lr, validation_set = None):
        """
        train_set: an iterable of x,y pairs, possibly batched
        lr: learning rate object 
        validation_set: an iterable of x,y pairs (possibly batched) for checking the accuracy
        """
        lr_gen = lr.lr_generator
        to = self.train_options
        curr_loss = 0.
        good = cases = st_cum = 0.
        start = time.time()
        for ep in range(to["epochs"]):
            for i, (x, y) in enumerate(iter(train_set)):
                st = next(lr_gen)
                st_cum += st
                y_hats = np.argmax(self.forward(x), axis=1)
                good += (y_hats==y).sum()
                cases += len(y)
                self.backward(y)
                self.update(st)
                curr_loss += self.get_loss()
                if (i+1) % to["report_interval"] == 0:
                    end = time.time()
                    report_validation = ""
                    if validation_set is not None:
                        val_good = val_cases = 0
                        for x,y in iter(validation_set):
                            y_hats = np.argmax(self.forward(x), axis=1)
                            val_good += (y_hats==y).sum()
                            val_cases += len(y)
                            report_validation = " dev acc {:.1%}, ".format(val_good/ val_cases)
                    report_header = "{}: epoch {} iter {} ({}):".format(time2str(end), ep, i, cases)
                    report_tr = " loss {:.4} tr_acc {:.1%}, st {:.3}".format(
                            curr_loss/cases, good/cases, st_cum / to['report_interval'])
                    print(report_header+report_validation+report_tr, flush=True)
                    start = time.time()
                    curr_loss = 0.
                    good = cases = 0.
                    st_cum = 0.
            lr.new_epoch()

    def to_pickle(self,filename):
        with open(filename,'wb') as p:
            pickle.dump(self,p)
