import numpy as np

vector_norm = lambda x: x/ np.linalg.norm(x,1)

transition_lambdas = vector_norm(np.array([3,2,1]))
period = "."