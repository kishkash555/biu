import numpy as np
from problem import problem
import geometries as gm


def split_solution(xs,N):
    speeds, deviations, multipliers = xs[:N], xs[N:2*N],xs[2*N:]
    return speeds, deviations, multipliers

class conjugate_gradient:
    def __init__(self,H,F, tol=1e-3):
        self.H = H
        # The canonical problem is x.T*H*x + F*x 
        # The solver solves x.T*H*x - F*x
        # Therefore we use the negative of F
        self.F = -F
        self.N = F.size
        self.tol = tol
    
    def solve(self,x0):
        x = x0.copy()
        norm = np.linalg.norm
        H, F, tol, N = self.H, self.F, self.tol, self.N
        b_hat = np.dot(H,x)
        if norm(b_hat-F) < tol:
            print("soltion found!")
            return x
        p = F-b_hat
        r = p.copy()
        for _ in range(4*N):
            alpha = np.dot(r.T,r)/np.dot(np.dot(p.T,H),p)
            x += alpha * p
            r_prev = r.copy()
            r -= alpha*np.dot(H,p)
            if norm(r) < tol:
                print("soltion found")
                return x
            beta = np.dot(r.T,r)/np.dot(r_prev.T, r_prev)
            p = r + beta*p

def make_problem():
    prob = problem(gm.compose_track(gm.turtulehead))
    prob.set_mu(0.7,1.2).set_top_speed().set_acc_and_brake_factors()
    prob.set_nominal_speed()
    return prob

def make_x0(prob, N):
    x0 = np.zeros(4*N-1)
    #x0 = np.concatenate([1/prob.initial_speed*np.ones(N), np.zeros(N)])
    return x0

def test_check_constraints_fulfilled():
    prob = make_problem()
    N = len(prob.segment_lengths)
    x0 = make_x0(prob, N) 
    prob.check_constrains_fulfilled(x0)



def test_solver():    
    prob = make_problem()
    N = len(prob.segment_lengths)
    x0 = make_x0(prob, N) 
    
    sigmas_d = 0.1*np.ones(N)
    H,F = prob.get_problem_matrices(sigmas_d)
    x_star = x0
    B=2*N
    while True:
        cj = conjugate_gradient(H,F)
        x_star = cj.solve(x_star)
        speed,deviat,mults = split_solution(x_star,N)
        n_mults = len(mults)
        if np.all(mults>=0):
            print("Solution found")
            break
        constraints_to_nullify = np.arange(n_mults)[mults<0]
        for i in constraints_to_nullify:
            H[B+i,:] = 0.
            H[:,B+i] = 0.
            F[B+i] = 0 
            x_star[B+i] = 0.
        1
    gm.plot_segments(prob.track_segments.segments,20,show=False,color='k')
    path = prob.track_segments.get_path_by_perturbations(deviat) 
    # gm.plot_segments(path.segments,20,show=True)
    1

if __name__ == "__main__":
    test_solver()

