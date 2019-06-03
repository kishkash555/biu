import numpy as np
from gazir import linear_layer, relu_layer, network


def test_linear():
    ll = linear_layer(4,2)
    x = np.random.rand(1,4)
    y = ll.forward(x)
    y1 = x.dot(ll.parameters['W']) + ll.parameters['b']
    y_str = np.array2string(y,precision=4)
    y1_str =  np.array2string(y1, precision=4)
    diff_str = np.array2string(y-y1, precision=6, suppress_small=True)
    print('{}\n \n- \n{} \n= \n{}'.format(y_str, y1_str, diff_str))
    assert(diff_str == '[[0. 0.]]')


def test_relu():
    x = np.random.rand(1,4)
    x[0,::2] = -x[0,::2]
    rl = relu_layer()
    y = rl.forward(x.copy())
    print('x={}\ny={}'.format(x,y))
    assert((np.abs(y[0,::2]) < 1e-9).all())    
    assert((y[0,1::2] == x[0,1::2]).all())    


def test_simple_network_fwd():
    x = np.random.rand(1,4)
    ll = linear_layer(4,2)
    net = network([ll, relu_layer()])
    y1 = net.forward(x.copy())
    ll.parameters['W'][:,0] = -ll.parameters['W'][:,0]
    ll.parameters['b'][:] = 0
    y2 = x.dot(ll.parameters['W']) + ll.parameters['b']
    y3 = net.forward(x.copy())
    s = lambda x: np.array2string(x,precision=4, suppress_small=True) 
#    W_str = np.array2string(ll.parameters[])
    print('x={}\ny1={}\ny2={}\ny3={}'.format(*map(s,[x,y1,y2,y3])))

    
if __name__ == "__main__":
    #test_linear()
    #test_relu()
    test_simple_network_fwd()