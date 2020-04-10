import geometries as gm
from problem import problem
import solver
import numpy as np


if __name__ == "__main__":
    rt = gm.compose_track()
    n_segments = rt.n_segments

    max_speed, cruise_speed = 80, 60
    pm = problem(rt).set_top_speed(max_speed).set_mu(0.8)

    cospeeds = np.ones(n_segments)/cruise_speed
    pert = (-np.ones(n_segments))**np.arange(n_segments) # alternating +1/-1

    H,F = pm.get_problem_matrices(np.ones(n_segments))
    cg = solver.conjugate_gradient(H,F,4*np.ones(n_segments))
    sol = cg.solve()
    
