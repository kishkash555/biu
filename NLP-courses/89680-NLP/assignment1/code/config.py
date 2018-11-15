import numpy as np


vector_norm = lambda x: x/ np.linalg.norm(x,1)

transition_lambdas = vector_norm(np.array([3,2,1]))
period = "."
start="*"

class frequncies:
    min_word_frequency = 6
    max_word_list_length = 150
    min_tag_frequency = 1e-3
    min_tag_pair_frequency = 5e-5