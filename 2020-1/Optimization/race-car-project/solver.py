import numpy as np
from problem import problem
import geometries as gm
import matplotlib.pyplot as plt

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
            # if alpha >0:
            #     x_bound = np.choose(p>0,lb,ub)
            # else: 
            #     x_bound = np.choose(p<0,lb,ub)
            # dist = np.abs(x-x_bound)
            # step_bound = np.abs(dist/p)
            # alpha = np.sign(alpha)*min(np.abs(alpha),step_bound.min())       

            x += alpha * p
            r_prev = r.copy()
            r -= alpha*np.dot(H,p)
            if norm(r) < tol:
                #print("soltion found")
                return x
            beta = np.dot(r.T,r)/np.dot(r_prev.T, r_prev)
            p = r + beta*p

def make_problem():
    prob = problem(gm.compose_track(gm.ushape))
    prob.set_mu(1.4,1.2).set_top_speed().set_acc_and_brake_factors().set_road_width()
    prob.set_nominal_speed()
    return prob

def make_x0(prob, N):
    x0 = np.zeros(prob.get_problem_dimension())
    #x0 = np.concatenate([1/prob.initial_speed*np.ones(N), np.zeros(N)])
    return x0

def test_check_constraints_fulfilled():
    prob = make_problem()
    N = len(prob.segment_lengths)
    x0 = make_x0(prob, N) 
    prob.check_constrains_fulfilled(x0)


def solve_using_boundaries(plot_solution=False):    
    prob = make_problem()
    prob_size = prob.get_problem_dimension()
    N = len(prob.segment_lengths)
    x0 = make_x0(prob, N) 
    tol = 1e-4
    sigmas_d = 0.01*np.ones(N)
    sigmas_u = 0.01*np.ones(N)
    boundary_d = np.zeros(N)
    boundary_u = np.zeros(N)
    H,F = prob.get_problem_matrices(sigmas_d, sigmas_u, boundary_d, boundary_u)
    Hg, Fg = prob.get_problem_matrices(sigmas_d, sigmas_u, goal_only=True)
    x_star = x0
    B=2*N
    #ub = np.concatenate([0.5*np.ones(N),4*np.ones(N),np.inf*np.ones(prob_size-2*N)])
    #lb = -ub
    constraints_to_nullify=np.zeros(prob_size-2*N,dtype=bool)
    while True:
        cj = conjugate_gradient(H,F)
        x_star = cj.solve(x_star)
        if x_star is None:
            print("h")
        speed,deviat,mults = split_solution(x_star,N)
        n_mults = len(mults)
        large_deviations = np.arange(N)[np.abs(deviat)>prob.road_width+tol]
        large_speeds = np.arange(N)[np.abs(speed)>0.5+tol]
        if np.all(mults>=-tol) and len(large_deviations)==0 and len(large_speeds)==0:
            break
        new_constraints_to_nullify = mults< -tol
        if sum(new_constraints_to_nullify)==0:
            if len(large_deviations)>len(large_speeds):
                boundary_d[large_deviations[0]] = np.sign(deviat[large_deviations[0]])  
            else:
                boundary_u[large_speeds[-1]] = np.sign(speed[large_speeds[-1]])
        H,F = prob.get_problem_matrices(sigmas_d, sigmas_u, boundary_d, boundary_u)
        constraints_to_nullify = np.logical_or(constraints_to_nullify,new_constraints_to_nullify)
        ind = np.arange(prob_size-2*N)[constraints_to_nullify]
        for i in ind:
            H[B+i,:] = 0.
            H[:,B+i] = 0.
            F[B+i] = 0 
            x_star[B+i] = 0.
        1
    me = np.array([prob.initial_speed]+list(prob.nominal_speed))
    t = 2*prob.segment_lengths/(me[:-1]+me[1:])
    delta_t = (x_star*(Hg.dot(x_star)+Fg))[:N]  
    if plot_solution:
        plt.subplot(121)
        gm.plot_segments(prob.track_segments.segments,20,show=False,color='k')
        path = prob.track_segments.get_path_by_perturbations(deviat*4) 
        gm.plot_segments(path.segments,3,show=False,color='b.')
        gm.plot_segments(path.segments,20,show=False,color='b--')
        plt.subplot(122)
        plt.plot([0]+list(np.cumsum(t)),me)

    return x_star, t, delta_t 

def solve_using_penalties():    
    prob = make_problem()
    prob_size = prob.get_problem_dimension()
    N = len(prob.segment_lengths)
    x0 = make_x0(prob, N) 
    
    sigmas_d = 0.01*np.ones(N)
    sigmas_u = 0.1*np.ones(N)
    H,F = prob.get_problem_matrices(sigmas_d, sigmas_u)
    x_star = x0
    B=2*N
    #ub = np.concatenate([0.5*np.ones(N),4*np.ones(N),np.inf*np.ones(prob_size-2*N)])
    #lb = -ub
    constraints_to_nullify=np.zeros(prob_size-2*N,dtype=bool)
    while True:
        cj = conjugate_gradient(H,F)
        x_star = cj.solve(x_star)
        speed,deviat,mults = split_solution(x_star,N)
        n_mults = len(mults)
        large_deviations = np.arange(N)[np.abs(deviat)>prob.road_width]
        large_speeds = np.arange(N)[np.abs(speed)>0.5]
        if np.all(mults>=0) and len(large_deviations)==0 and len(large_speeds)==0:
            # print("Solution found. speed {} deviations {}".format(
            #     np.array2string(prob.nominal_speed*(1+speed)*3.6,precision=1,suppress_small=True),
            #     np.array2string(deviat,precision=2,suppress_small=True),
            # ))
            break
        new_constraints_to_nullify = mults<0
        if sum(new_constraints_to_nullify)==0:
            # print("large dev {} large speed {}".format(large_deviations, large_speeds))
            sigmas_d[large_deviations] *= 1.25
            sigmas_u[large_speeds] *=1.25
        # else:
        #     print("nullify", np.arange(prob_size-2*N)[new_constraints_to_nullify])
        H,F = prob.get_problem_matrices(sigmas_d, sigmas_u)
        constraints_to_nullify = np.logical_or(constraints_to_nullify,new_constraints_to_nullify)
        ind = np.arange(prob_size-2*N)[constraints_to_nullify]
        for i in ind:
            H[B+i,:] = 0.
            H[:,B+i] = 0.
            F[B+i] = 0 
            x_star[B+i] = 0.
        1
    gm.plot_segments(prob.track_segments.segments,20,show=False,color='k')
    path = prob.track_segments.get_path_by_perturbations(deviat*4) 
    gm.plot_segments(path.segments,3,show=False,color='b.')
    gm.plot_segments(path.segments,20,show=True,color='b--')
    1

if __name__ == "__main__":
    solve_using_boundaries(True)
    plt.show()
    
