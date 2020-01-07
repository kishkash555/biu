import numpy as np

class kalman_simple:
    def __init__(self,a22,q22,R,x0):
        self.a22 = a22
        self.q22 = q22
        self.R = R
        self.x0 = x0

    def simulate(self,nsteps=100):
        A, C, Q, R = self.A, self.C, self.Q, self.R
        x_i = self.x0
        x = [x_i]
        for s in range(nsteps):
            x_i = A.matmul(x_i)
