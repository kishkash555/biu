import numpy as np
from loglinear import softmax, create_classifier, loss_and_gradients
from grad_check import gradient_check


def softmax_sanity():
    # if __name__ == '__main__':
    # Sanity checks for softmax. If these fail, your softmax is definitely wrong.
    # If these pass, it may or may not be correct.
    print("running softmax tests")
    test1 = softmax(np.array([1,2]))
    test2 = softmax(np.array([1001,1002]))
    test3 = softmax(np.array([-1001,-1002])) 
    print ("test1: {}".format(test1))
    assert np.amax(np.fabs(test1 - np.array([0.26894142,  0.73105858]))) <= 1e-6
    print ("test2: {}".format(test2))
    assert np.amax(np.fabs(test2 - np.array( [0.26894142, 0.73105858]))) <= 1e-6
    print ("test3: {}".format(test3))
    assert np.amax(np.fabs(test3 - np.array([0.73105858, 0.26894142]))) <= 1e-6
    print("softmax tests passed")

def grad_sanity():
    # Sanity checks. If these fail, your gradient calculation is definitely wrong.
    # If they pass, it is likely, but not certainly, correct.
    # import sys
    #sys.path.append("C:\Shahar\BarIlan\NLP-courses\89687-DL\Assignment1\code\loglinear.py")
    #print(sys.path)
    #from .grad_check import gradient_check
    global W,b
    W,b = create_classifier(3,6)

    def _loss_and_W_grad(W):
        global b
        x = np.array([[1,2,3]],np.double)
        loss,grads = loss_and_gradients(x,0,[W,b])
        return loss,grads[0]

    
    def _loss_and_b_grad(b):
        global W
        x = np.array([[1,2,3]],np.double)
        loss,grads = loss_and_gradients(x,0,[W,b])
        return loss,grads[1]
 
    for _ in range(10):
        W = np.random.randn(W.shape[0],W.shape[1])
        b = np.random.randn(b.shape[0],b.shape[1])
        gradient_check(_loss_and_b_grad, b)
        gradient_check(_loss_and_W_grad, W)


    
if __name__ == "__main__":
    grad_sanity()