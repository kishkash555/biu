import numpy as np
import loglinear as ll

STUDENT={'name': 'SHAHAR SIEGMAN',
         'ID': '011862141'}

def layer_output(x,params):
    # calculates the output of a layer of the form
    # a = tanh(z) = tanh(xW+b)
    W, b = params
    a = np.tanh(np.dot(x,W)+b)
    return a

def classifier_output(x, params):
    # calculation is done in 2 stages
    # we need to feed-forward from x to x1
    # then solve x1 with loglinear.predict
    W, b, U, b_prime = params
    x1 = layer_output(x, (W,b))
    probs = ll.classifier_output(x1,(U,b_prime))
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """

    W, b, U, b_tag = params
    out_dim = U.shape[1]
    # first, do the entire feedforward
    z2 = np.dot(x,W) + b
    a2 = np.tanh(z2)
    z3 = np.dot(a2,U) + b_tag
    y_hat = ll.softmax(z3)

    y_e = ll.to_one_hot_row(y,out_dim)

    y_diff = y_hat - y_e

    # now we can use backprop
    gU = np.dot(a2.transpose(), y_diff )
    gb_tag = y_diff.copy()

    delta2 = np.dot(y_diff, U.T) 
    delta2 = delta2 * (1-a2 ** 2)

    gW = np.dot(x.transpose(), delta2)
    gb = delta2

    loss = ll.logloss(y_e, y_hat)
    return loss,[gW, gb, gU, gb_tag]

    

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.zeros((in_dim,hid_dim), np.double)
    b = np.zeros((1,hid_dim), np.double)
    U = np.zeros((hid_dim,out_dim), np.double)
    b_tag = np.zeros((1,out_dim), np.double)
    return W,b,U, b_tag

