import numpy as np

STUDENT={'name': 'Shahar Siegman',
         'ID': '011862141'}

def gradient_check(f, x):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 
    fx, grad = f(x) # Evaluate function value at original point
    if not np.all(grad.shape==x.shape):
        print("the shapes of x ({}) and the gradient ({}) do not match ".format(x.shape, grad.shape))
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        ### modify x[ix] with h defined above to compute the numerical gradient.
        ### if you change x, make sure to return it back to its original state for the next iteration.
        v = x[ix] 
        x[ix] = v + h/2
        f2,_ = f(x)
        x[ix] = v - h/2
        f1,_ = f(x)
        x[ix] = v
        numeric_gradient = (f2[0]-f1[0])/h
        
        # print("f1: {}, f2: {}, numeric: {}, analytic: {}, location: {}, ".\
        #     format(f1,f2, numeric_gradient, grad[ix], ix ))
        # Compare gradients
        reldiff = abs(numeric_gradient - grad[ix]) / max(1, abs(numeric_gradient), abs(grad[ix]))
        if reldiff > 1e-5:
            #print("Gradient check failed.")
            print("gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numeric_gradient))
        #else: 
        #    print("Gradient ok. numeric {} analytic {}".format(numeric_gradient, grad[ix]))
        it.iternext() # Step to next index
    print("Gradient check complete ")

def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradient_check(quad, np.array(123.456))      # scalar test
    x = np.random.randn(3,)
    gradient_check(quad, x)    # 1-D test
    gradient_check(quad, np.random.randn(4,5))   # 2-D test
    print("")

if __name__ == '__main__':
    # If these fail, your code is definitely wrong.
    sanity_check()    
    