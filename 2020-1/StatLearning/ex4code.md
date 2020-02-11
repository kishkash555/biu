```python
import numpy as np

np.random.seed(1235)

class kalman_simple:
    """
    a kalman model with the following hard-coded constants - makes the simulation much easier
    A = [[1,1],[0, a22]] so only 1 free parameter
    C = [1,0] only the 1st coordinate is reflected in y
    Q = [[0,0],[0,q22]] this means the process noise is only in the 2nd coordinate 
    """
    def __init__(self,a22,q22,R,S0):
        x0 = np.array([[S0*np.random.randn(),0]]).T
        self.S0 = S0
        self.a22 = a22
        self.q22 = q22
        self.R = R
        self.x0 = x0
        self.step = 0
        self.x = np.zeros((2,0),dtype=float)
        self.y = np.zeros(0,dtype=float)
        

    def simulate(self,nsteps=100):
        a22, q22, R = self.a22, self.q22, self.R
        A = np.array([[1,1],[0, a22]])
        x_t = self.x0
        x = []
        y = []
        for _ in range(nsteps):
            x_t = np.matmul(A, x_t)
            x_t[1] += q22*np.random.randn()
            y_t = x_t[0] + R * np.random.randn()
            x.append(x_t)
            y.append(y_t)
        self.step += nsteps
        self.x = np.hstack([self.x, np.hstack(x)])
        self.y = np.hstack([self.y, np.hstack(y)])

    def kalman_filter(self):
        p = np.zeros((2,2))
        p[1,1] = self.S0
        x = np.zeros((2,1))
        
        a22, q22, R = self.a22, self.q22, self.R
        A = np.array([[1,1],[0, a22]])
        Q = np.array([[1,1],[0, q22]])
        C = np.array([[1.,0.]])
        x_hat = []
        for s in range(self.step):
            x = np.matmul(A,x)
            innov = self.y[s]- x[0]
            p = np.matmul(np.matmul(A,p),A.T)+Q
            k = p[:,0]/(p[0,0]+R)
            k = k[:,np.newaxis]
            i_ktc = np.eye(2)- np.matmul(k,C)
            krk = R*np.matmul(k,k.T)
            p = np.matmul(np.matmul(i_ktc, p), i_ktc.T) + krk
            x = x + k*innov
            x_hat.append(x)
        self.x_hat = np.hstack(x_hat)
```