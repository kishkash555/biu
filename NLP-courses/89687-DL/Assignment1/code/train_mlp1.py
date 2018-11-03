import mlp1 as lp
import train_loglin as tl
import random
import utils 
import config 
import vectorize_utils
import numpy as np

from collections import Counter

STUDENT={'name': 'SHAHAR SIEGMAN',
         'ID': '011862141'}



def accuracy_on_dataset(dataset, params):
    y_y_hat = [(y, lp.predict(x, params)) for x,y in dataset]
    print("yhat counter: {}".format(Counter([x[1] for x in y_y_hat])))
    is_good = [a==b for a, b in y_y_hat]
    return sum(is_good)/len(is_good)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for x, y in train_data:
            x = np.array(x, ndmin = 2, dtype=np.double) # make row vector    
            loss, grads = lp.loss_and_gradients(x,y,params)
            cum_loss += loss
            for i in range(len(params)):
                params[i] = params[i] - learning_rate * grads[i]

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print("I {0:}, train_loss {1:}, train_accuracy {2:0.5f}, dev_accuracy {3:0.5f}"\
            .format(I, train_loss, train_accuracy, dev_accuracy))
    return params


def main(text_to_ngram):
    train_data = utils.read_data(config.filename_train)
    symbol_dict = tl.initialize_symbol_dict(train_data)
    label_dict = tl.initialize_label_dict(train_data)
    xy_train = list(tl.xy_generator(train_data, text_to_ngram, symbol_dict, label_dict))

    dev_data = utils.read_data(config.filename_dev)
    xy_dev = list(tl.xy_generator(dev_data,text_to_ngram, symbol_dict, label_dict))
    in_dim = min(config.max_count, len(symbol_dict))
    out_dim = len(label_dict)
    hidden_dim = config.mlp1.hidden_layer_size
    print("problem dimensions are: {}".format((in_dim, hidden_dim, out_dim)))
    params = lp.create_classifier(in_dim, hidden_dim,out_dim)
    params = [randomize_array(p) for p in params]
    trained_params = train_classifier(xy_train, xy_dev, config.mlp1.num_iterations, config.mlp1.learning_rate, params)
    return trained_params

def randomize_array(m):
    return np.random.randn(*m.shape)


if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.    
    trained_params = main(utils.text_to_bigrams)
    

