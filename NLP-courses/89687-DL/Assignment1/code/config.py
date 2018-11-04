# parameters and configurations

#from os import path
# filename_train = path.join("data","train")
# filename_dev = path.join("data","dev")
# filename_test = path.join("data","test")

filename_train = "train"
filename_dev = "dev"
filename_test = "test"

max_count = 600

debug = False

class train_filter:
    hash = True
    at_sign = True
    short_strings = 8

class loglin:
    num_iterations = 20
    learning_rate = 0.05

class mlp1:
    num_iterations = 20
    learning_rate = 0.02
    hidden_layer_size = 150
    seed = 335

class mlpn:
    num_iterations = 5
    learning_rate = 0.02
    layer_sizes = [600, 150, 6]
    seed = 335