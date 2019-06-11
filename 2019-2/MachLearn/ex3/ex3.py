import numpy as np
from gazir import linear_layer, relu_layer, network, softmax_nll_layer
from data_iterator import data_iterator
from learn_rate import learn_rate_schedule
input_len = 784
output_len =10
train_x_file = 'train_x.npy'
train_y_file = 'train_y'
expected_train_size= 55000
validation_sample_ratio = 11
train_batch_size = 64
validation_batch_size = 128

def load_data():
    data_x = np.load(train_x_file)
    data_y = np.loadtxt(train_y_file,dtype=int)
    if data_x.shape[0] != expected_train_size:
        raise ValueError("expected {} rows in train data, found {}".format(expected_train_size, data_x.shape[0]))
    if data_y.shape[0] != expected_train_size:
        raise ValueError("number of labels, {}, does not match number of train data {}".format(data_y.shape[0], expected_train_size))

    validation_x = data_x[::validation_sample_ratio]
    validation_y = data_y[::validation_sample_ratio]
    train_x = np.delete(data_x, list(range(0, data_x.shape[0], validation_sample_ratio)), axis=0)
    train_y = np.delete(data_y, list(range(0, data_y.shape[0], validation_sample_ratio)), axis=0)
    # del data_x
    di_train = data_iterator(train_x, train_y, train_batch_size)
    di_valid = data_iterator(validation_x, validation_y, validation_batch_size, False)
    return di_train, di_valid

def create_network():
    return network([linear_layer(input_len,100), relu_layer(), linear_layer(100,output_len), softmax_nll_layer()])
    #return network([linear_layer(input_len, output_len),  softmax_nll_layer()])

def ex3_main():
    net = create_network()
    di_train, di_valid = load_data()
    lr = learn_rate_schedule('constant',momentum=True, eta=0.001, alpha=10, gamma=0)
    net.set_train_options(epochs=50, report_interval=250)
    net.train(di_train, lr, di_valid)

if __name__ == "__main__":
    ex3_main()
    