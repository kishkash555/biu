import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    z_and_a = feedforward_loop(x, params)
    return z_and_a[-1][1]

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def layer_ff(a_in, layer_params, sigma):
    """
    single stage of the feed-forward technique of MLP
    computes the input of the layer z = a_in*W + b
    and the output of the layer a_out = tanh(z)
    a_in and a_out are row vectors, W is a matrix
    sigma is a function from a (row) array to a (row) array
    which calculates the nonlinearity
    """
    
    W, b = layer_params
    #print("in layer ff. a_in {}, W {}, b {}".format(a_in.shape,W.shape, b.shape))
    if type(a_in) != np.ndarray:
        print("layer_ff got a_in that is not a numpy array")
        raise AssertionError
    z = np.dot(a_in, W) + b
    a_out = sigma(z)
    return (z, a_out)

def layer_backpropagate(delta_in, a_previous, a_current, layer_params,sigma_prime):
    """
    single stage of the backpropagation technique of MLP
    computes the derivative of the layer's params and the delta to be used in the previous layer
    sigma_prime is a function from a (row) array to a (row) array
    which calculates the deirvative of the nonlinearity
    """
    W, b = layer_params
    delta_out = np.dot(delta_in, W.T)
    #print("delta_out: {}, delta_in: {}, W: {}, a_prev:{}, a_cur:{}"\
    #    .format(delta_out.shape, delta_in.shape, W.shape,a_previous.shape, a_current.shape))
    delta_out = delta_out * sigma_prime(a_current)
    gW = np.dot(a_previous.transpose(), delta_out)
    gb = delta_out

    return  delta_out, gW, gb


def feedforward_loop(x, params):
    z_and_a = [(np.array([]),x)] # 
    z_and_a.append(layer_ff(x,params[0:2],np.tanh))
    curr_layer = 1 # layers are numbered from 0 to layers-1
    layers = int(len(params)/2)
    # feedforward loop of hidden layer
    while curr_layer  < layers-1: # the last two parameters are for the softmax layer
        a_in = z_and_a[-1][1]
        z_and_a.append(layer_ff(a_in, params[curr_layer*2:(curr_layer*2+2)],np.tanh))
        curr_layer += 1
    # now feedforward the softmax
    z_and_a.append(layer_ff(z_and_a[-1][1],params[-2:],ll.softmax))
    return z_and_a



def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """

    if len(x.shape)==1:
        x = np.array(x, ndmin=2)
    z_and_a = feedforward_loop(x,params)   
    # backpropagate output layer
    # given 3 layers, numbered 0, 1, 2 the z_and_a indexes correspond 
    # to the layers as follows:
    # z_and_a[0] => layer[0], z_and_a[1] => layer[1], z_and_a[2] => layer[2]
    # the 0th layer has nothing in the z and x as the a
    curr_layer = int(len(params)/2)-1  # last layer that has a matrix 
    y_hat  = z_and_a[curr_layer +1][1] # the final activation is the estimated y
    y_e = ll.to_one_hot_row(y,y_hat.shape[1])
    delta = y_hat - y_e
    grads = [np.dot(z_and_a[curr_layer][1].transpose(), delta), delta]
    
    curr_layer -= 1
    # now apply backpropagation to the tanh layers
    while curr_layer >= 0:
        delta, gW, gb = layer_backpropagate(
            delta,
            z_and_a[curr_layer][1],
            z_and_a[curr_layer+1][1],
            params[(curr_layer+1)*2:(curr_layer+1)*2+2],
            lambda a: 1- a**2
            )
        grads = [gW, gb] + grads
        curr_layer -= 1

    loss = ll.logloss(y_e, y_hat)
    return loss, grads

        
def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """

    Ws = [np.zeros((rows, cols),np.double) for rows, cols in zip(dims[:-1],dims[1:])]
    bs = [np.zeros((1, cols),np.double) for cols in dims[1:]]
    params = []
    for W, b in zip(Ws,bs):
        params.append(W)
        params.append(b)
    return params

