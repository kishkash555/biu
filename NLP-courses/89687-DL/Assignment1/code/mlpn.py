import numpy as np
import loglinear as ll

STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    return probs

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
    z = a_in * W + b
    a_out = sigma(z)
    return z, a_out

def layer_backpropagate(delta_in, a_previous, a_current, layer_params,sigma_prime):
    """
    single stage of the backpropagation technique of MLP
    computes the derivative of the layer's params and the delta to be used in the previous layer
    sigma_prime is a function from a (row) array to a (row) array
    which calculates the deirvative of the nonlinearity
    """
    W, b = layer_params
    delta_out = np.dot(delta_in, W.T)
    delta_out = delta_out * sigma_prime(a_current)

    gW = np.dot(a_previous, delta_out)
    gb = delta_out

    return  delta_out, gW, gb




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
    z_and_a = []

    z_and_a.append(layer_ff(x,params[0:2],np.tanh))
    curr_layer = 1
    # feedforward loop of hidden layer
    while curr_layer *2 < len(params):
        a_in = z_and_a[-1]
        z_and_a.append(layer_ff(a_in, params[curr_layer*2:(curr_layer*2+2)],np.tanh))
        curr_layer+=1

    # feedforward output layer
    z_and_a.append(layer_ff(z_and_a[-1],params[-2:],ll.softmax))
    y_hat  = z_and_a[-1][1] # the final activation is the estimated y
    delta = y-y_hat
 
    # now apply backpropagation
    while curr_layer > 0:
        layer_backpropagate(delta,)
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

