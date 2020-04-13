import numpy as np
from problem import problem
import geometries as gm


def split_solution(x_star):
    N = len(x_star)//2
    return 1/x_star[:N], x_star[N:]

class conjugate_gradient:
    def __init__(self,H,F,boundaries, tol=1e-3):
        self.H = H
        # The canonical problem is x.T*H*x + F*x 
        # The solver solves x.T*H*x - F*x
        # Therefore we use the negative of F
        self.F = -F
        self.b = boundaries
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
        for _ in range(N):
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
    prob.set_mu(0.5).set_top_speed().set_acc_and_brake_factors()
    return prob

def make_x0(prob, N):
    x0 = np.concatenate([1/prob.initial_speed*np.ones(N), np.zeros(N)])
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
    
    sigmas_centrip = np.zeros(N*2)

    sigmas_acc = 1*np.ones(N)
    sigmas_br = 0.1*np.ones(N)
    sigmas_width = 0.1*np.ones(N)
    sigmas_top_speed = 0.1*np.ones(N)
    H,F = prob.get_problem_matrices(sigmas_centrip, sigmas_acc, sigmas_br, sigmas_width, sigmas_top_speed)

     
    cj = conjugate_gradient(H,F,4)
    x_star = cj.solve(x0)
    speed,drift = split_solution(x_star)
    gm.plot_segments(prob.track_segments.segments,20,show=False,color='k')
    path = prob.track_segments.get_path_by_perturbations(drift) 
    # gm.plot_segments(path.segments,20,show=True)
    1

if __name__ == "__main__":
    test_solver()

