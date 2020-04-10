import numpy as np

class conjugate_gradient:
    def __init__(self,H,F,boundaries, tol=1e-3):
        self.H = H
        self.F = F
        self.b = boundaries
        self.N = F.size
        self.tol = tol
    
    def solve(self,x0):
        x = x0.copy()
        norm = np.linalg.norm
        H, F, tol, N = self.H, self.F, self.tol, self.N
        b_hat = np.dot(H,x)
        if norm(b_hat-b) < tol:
            return x
        p = b-b_hat
        r = p
        for _ in range(N):
            alpha = np.dot(r.T,r)/np.dot(np.dot(p.T,H),p)
            x += alpha * p
            r_prev = r.copy()
            r -= alpha*np.dot(H,p)
            if norm(r) < tol:
                print("soltion found")
                break
            beta = np.dot(r.T,r)/np.dot(r_prev.T, r_prev)
            p = r + beta*p



