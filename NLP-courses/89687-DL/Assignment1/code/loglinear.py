import numpy as np
import config

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

def to_one_hot_row(index,n_ele):
    """
    """
    ret = np.zeros((1,n_ele))
    ret[0,index] = 1
    return ret

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
    shape_W = W.shape
    out_dim = shape_W[1]
    y_e = to_one_hot_row(y,out_dim) 

    if type(x)==list:
        x = np.array(x,np.double,ndmin=2)
    shape_x = x.shape
    if len(shape_x)==1 or shape_x[0] != 1:
        print("x should be a row vector, actual shape: {}".format(shape_x))
        raise AssertionError
    in_dim = shape_x[1]
    
    if shape_W[0] != in_dim:
        print("number of rows of W ({}) mismatches length of X ({})".format(shape_W[0],in_dim))
        raise AssertionError
    out_dim = shape_W[1]
    shape_b = b.shape
    if len(shape_b)==1:
        if config.debug:
            print("b should be a 2d array, actual shape: {}".format(shape_b))
        b = np.array(b,np.double, ndmin=2)

    elif shape_b[0] != 1:
        print("b should be a row vector, actual shape: {}".format(shape_b))
        raise AssertionError
    elif shape_b[1] != out_dim:
        print("length of b ({}) mismatches columns of W ({})".format(shape_b[1],out_dim))

    y_hat = classifier_output(x,params)
    loss = logloss(y_e, y_hat)
    y_diff = y_hat-y_e
    gW = np.dot(x.transpose(),y_diff)
    gb = y_diff
    if not np.all(gW.shape==W.shape):
        print("problem with calculation of gW, size: {}, expected: {}".format(gW.shape, W.shape))
        print("shape y_diff: {}, shape x: {}".format(y_diff.shape, np.matrix(x).shape))
        raise AssertionError
    return loss,[gW,gb]

def logloss(y, y_hat):
    return -np.dot(y,np.log(y_hat).transpose())

def create_classifier(in_dim, out_dim):
    """
    returns the parameters (W,b) for a log-linear classifier
    with input dimension in_dim and output dimension out_dim.
    """
    W = np.zeros((in_dim, out_dim))
    b = np.zeros(out_dim)
    return [W,b]


if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    test1 = softmax(np.array([1,2]))
    print(test1)    
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([1001,1002]))
    print(test2)
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6

    test3 = softmax(np.array([-1001,-1002])) 
    print(test3) 
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6


    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    from grad_check import gradient_check

    W,b = create_classifier(3,4)

    def _loss_and_W_grad(W):
        global b
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss, grads[0]

    def _loss_and_b_grad(b):
        global W
        loss,grads = loss_and_gradients([1,2,3],0,[W,b])
        return loss, grads[1]

    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0])
        # print("b: {}".format(b))
        # print("itreation: {} grad check b".format(_))
        gradient_check(_loss_and_b_grad, b)
        # print("grad check W")
        gradient_check(_loss_and_W_grad, W)
