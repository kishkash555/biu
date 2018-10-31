import loglinear as ll
import random
import utils 
import config 
import vectorize_utils

from collections import Counter

STUDENT={'name': 'SHAHAR SIEGMAN',
         'ID': '011862141'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return None

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        pass
    return good / (good + bad)

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
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            for param, grad in zip(params, grads):
                param = param - learning_rate * grad

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(I, train_loss, train_accuracy, dev_accuracy)
    return params

def initialize_symbol_dict(train_data):
    all_bigrams = utils.create_bigram_vocab(train_data,config.max_count)
    symbol_dict = vectorize_utils.create_symbol_dict(all_bigrams)
    return symbol_dict

def initialize_label_dict(train_data):
    all_labels = set([lb for lb, _ in train_data])
    label_dict = vectorize_utils.create_symbol_dict(all_labels)
    return label_dict

def xy_generator(corpus, text_to_ngram, symbol_dict, label_dict):
    ny = len(label_dict)
    nx = len(symbol_dict)
    for label, text in corpus:
        ngrams = Counter(text_to_ngram(text)).most_common()
        y = label_dict[label] 
        x = vectorize_utils.generate_vector(nx,ngrams,symbol_dict)
        yield x, y

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    # ...
    
    train_data = utils.read_data(config.filename_train)
    symbol_dict = initialize_symbol_dict(train_data)
    label_dict = initialize_label_dict(train_data)
    xy = xy_generator(train_data, utils.text_to_bigrams, symbol_dict, label_dict)
    in_dim = min(config.max_count, len(label_dict))
    out_dim = len(label_dict)
    print("problem dimensions are: {}".format((in_dim, out_dim)))
    params = ll.create_classifier(in_dim, out_dim)
    #trained_params = train_classifier(train_data, dev_data, num_iterations, learning_rate, params)



