import numpy as np

STUDENT={'name': 'Shahar Siegman',
         'ID': '011862141'}

def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    x0 = x-x.mean()
    ret = np.exp(x0)
    return ret / ret.sum()
    # YOUR CODE HERE
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    # return x
    ### why does the original code say return x?

def classifier_output(x, params):
    """
    Return the output layer (class probabilities) 
    of a log-linear classifier with given params on input x.
    """
    W,b = params
    # we use Z = xW + b, where x and b are row vectors
    f_at_x = np.dot(x,W) + b
    probs = softmax(f_at_x)
    return probs

def predict(x, params):
    """
    Returnss the prediction (highest scoring class id) of a
    a log-linear classifier with given parameters on input x.

    params: a list of the form [(W, b)]
    W: matrix
    b: vector
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W,b = params
    # todo: check the shape of b
    shape_x = x.shape
    if len(shape_x)>1 and shape_x[1] != 1:
        print("x should be a column vector, actual shape: {}".format(shape_x))
        raise AssertionError
    in_dim = shape_x[0]
    shape_W = W.shape
    if shape_W[0] != in_dim:
        print("number of rows of W ({}) mismatches length of X ({})".format(shape_W[0],in_dim))
        raise AssertionError
    out_dim = shape_W[1]
    shape_b = b.shape
    if len(shape_b)>1 and shape_b[1] != 1:
        print("b is not a row vector, actual shape: {}".format(shape_b))
        raise AssertionError
    if shape_b[0] != out_dim:
        print("length of b ({}) mismatches columns of W ({})".format(shape_b[1],out_dim))

    y_hat = classifier_output(x,params)
    print("y_hat: {}".format(y_hat))
    loss = logloss(y, y_hat)
    y_diff = np.matrix(y_hat-y)
    gW = np.dot(np.matrix(x).transpose(),y_diff)
    gb = y_diff 
    if not np.all(gW.shape==W.shape):
        print("problem with calculation of gW, size: {}, expected: {}".format(gW.shape, W.shape))
        print("shape y_diff: {}, shape x: {}".format(y_diff.shape, np.matrix(x).shape))
        raise AssertionError
    return loss,[gW,gb]

def logloss(y, y_hat):
    print("in log loss. shape y: {}, shape y_hat: {}".format(y.shape, y_hat.shape))
    return np.dot(y,np.log(y_hat))

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]

