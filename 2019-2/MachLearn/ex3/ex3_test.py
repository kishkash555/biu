import numpy as np
from gazir import linear_layer, relu_layer, network, softmax_nll_layer
from data_iterator import data_iterator
from learn_rate import learn_rate_schedule
s = lambda x: np.array2string(x,precision=4, suppress_small=True) 

def test_linear():
    ll = linear_layer(4,2)
    x = np.random.rand(1,4)
    y = ll.forward(x)
    y1 = x.dot(ll.parameters['W']) + ll.parameters['b']
    diff_str = np.array2string(y-y1, precision=6, suppress_small=True)
    print('{}\n \n- \n{} \n= \n{}'.format(s(y), s(y1), diff_str))
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
    ll.parameters['W'] = np.abs(ll.parameters['W'])
    ll.parameters['W'][:,0] = -ll.parameters['W'][:,0]
    ll.parameters['b'][:] = 0
    y2 = x.dot(ll.parameters['W']) + ll.parameters['b']
    y3 = net.forward(x.copy())
#    W_str = np.array2string(ll.parameters[])
    print('x={}\ny2={}\ny3={}'.format(*map(s,[x,y2,y3])))
    assert(np.abs(y3[0,0]) < 1e-9)
    assert(np.abs(y3[0,1] - y2[0,1])< 1e-9)    

def test_softmax_network_fwd():
    x = np.random.rand(1,4)
    ll = linear_layer(4,2)
    net = network([ll, softmax_nll_layer()])
    y1 = net.forward(x.copy())
    print('x={}\ny1={}'.format(s(x),s(y1)))
    assert(abs(y1.sum()-1) < 1e-9)
    assert((np.argsort(y1) == np.argsort(ll.forward(x))).all())

def test_softmax_network_bkwd():
    x1 = np.random.rand(1,4)
    net = network([softmax_nll_layer()])
    y1 = net.forward(x1.copy())
    dx = net.backward(0)
    for i in range(dx.shape[0]):
        x2 = x1.copy()
        x2[0,i] = x2[0,i] + 1e-4 
        y2 = net.forward(x2.copy())
        print(np.log(y1[0,0])-np.log(y2[0,0]), dx[i]*1e-4)
#    print("{}\n{}\n{}".format(s(x1),s(x2),))

def create_linear_data(dim=5, n_train=50, cube_side=10.):
    angle = np.random.rand(1,dim-1) * np.pi
    angle[-1] = angle[-1]*2 # last angle has range [0,2*pi)
    w = angles_to_unit_vector(angle)
    bias = np.random.rand(1)*(cube_side/2.5)-(cube_side/5)
    train_x = np.random.rand(n_train,dim) * cube_side - (cube_side/2)
    train_y = (0.5 * np.sign(train_x.dot(w.T) + bias) + 1).astype(int)
    return train_x, train_y


def test_train_network():
    dim = 5
    train_x, train_y = create_linear_data(dim, n_train=100)
    train_y[::7] = np.mod(train_y[::7]+1,2)
    di = data_iterator(train_x, train_y, 1)
    net = network([linear_layer(dim,2),softmax_nll_layer()])
    lr = learn_rate_schedule('exponential',eta=0.1, alpha=0.5).set_step_width('epoch')
    net.train(di, lr)


def angles_to_unit_vector(angles):
    """
    rows: separate vectors
    columns: angles in different dimensions
    """
    dims = angles.shape[1] + 1
    ret = np.ones((angles.shape[0], dims))
    for i in range(dims-1):
        ret[:,i] *= np.cos(angles[:,i])
        ret[:,i+1:] *= np.sin(angles[:,i])
    return ret

def test_exp_step():
    lr = learn_rate_schedule('exponential', eta=0.1, alpha=0.5).set_step_width('epoch')
    lrg = lr.lr_generator
    for ep in range(3):
        for _ in range(5):
            print(next(lrg))
        print()
        lr.new_epoch()

if __name__ == "__main__":
    #test_linear()
    #test_relu()
    #test_simple_network_fwd()
    #test_softmax_network_fwd()
    #test_softmax_network_bkwd()
    test_train_network()
    #test_exp_step()