# parameters and configurations
#classifier_dimensions = (600,12)
from os import path
filename_train = path.join("data","train")
filename_dev = path.join("data","dev")
filename_test = path.join("data","test")
max_count = 600

class loglin:
    num_iterations = 20
    learning_rate = 0.05

class mlp1:
    num_iterations = 20
    learning_rate = 0.02
    hidden_layer_size = 150


class mlpn:
    num_iterations = 20
    learning_rate = 0.02
    layer_sizes = [600, 150, 6]