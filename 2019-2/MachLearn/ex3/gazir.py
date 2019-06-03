import numpy as np
from scipy.special import softmax
from collections import OrderedDict

class layer:
    def __init__(self):
        self.ctx = {}

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, grad_output):
        raise NotImplementedError()
    
    def loss(self, y):
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
        self.ctx['input'] = x
        return x.dot(W) + b

    def backward(self, grad_output):
        W, _ = self.parameters.values()
        x = self.ctx['input']
        grad_input = grad_output.dot(W)
        grad_weight = grad_output.T.dot(x)
        grad_bias = grad_output
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
        super().__init__(self)
    
    def forward(self, x):
        ret = softmax(x)
        self.ctx['softmax'] = ret.copy()
        return ret

    def backward(self, grad_output): 
        pass
    
    def loss(self, y):
        ret = -self.ctx['softmax']
        ret[y] = ret[y] + 1
        return ret



class network(layer):
    def __init__(self, layer_obj, layer_names = None):
        super().__init__()
        layer_names = layer_names or ['layer{:02}'.format(i) for i in range(len(layer_obj))]
        self.layers = OrderedDict(zip(layer_names, layer_obj))
    
    def forward(self, x):
        z = x
        for layer_name, layer_obj in self.layers.items():
            z = layer_obj.forward(z)
        return z

    def backward(self, y):
        is_last_layer = True
        for layer_name, layer_obj in reversed(self.layers.items()):
            if is_last_layer:
                grad = layer_obj.loss(y)
                is_last_layer = False
            else:
                grad = layer_obj.backward(grad)

    def update(self, stepsize):
        for layer_name, layer_obj in self.layers.items():
            layer_obj.update(stepsize)


