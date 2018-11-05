import loglinear as ll
import random
import utils 
import config 
import vectorize_utils
import numpy as np

from collections import Counter

STUDENT={'name': 'SHAHAR SIEGMAN',
         'ID': '011862141'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    return None

def accuracy_on_dataset(dataset, params):
    y_y_hat = [(y, ll.predict(x, params)) for x,y in dataset]
    if config.debug:
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
    prev_dev_accuracy = 0
    for I in range(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        random.shuffle(train_data)
        for x, y in train_data:
            x = np.array(x, ndmin = 2) # make row vector    
            loss, grads = ll.loss_and_gradients(x,y,params)
            cum_loss += loss
            params[0] = params[0] - learning_rate * grads[0]
            params[1] = params[1] - learning_rate * grads[1]
            #for param, grad in zip(params, grads):
            #    param = param - learning_rate * grad

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        if dev_accuracy < prev_dev_accuracy and I > config.loglin.min_iterations:
            print("early stopping criterion in iteration {} - detriorating dev accuracy".format(I))
            params = prev_params
            break
        prev_params = [p.copy() for p in params]
        prev_dev_accuracy = dev_accuracy
        print("I {}, train_loss {}, train_accuracy {}, dev_accuracy {}"\
                .format(I, loss, train_accuracy, dev_accuracy))
    return params

def initialize_symbol_dict(train_data, text_to_ngram):
    all_bigrams = utils.create_ngram_vocab(train_data,config.max_count, text_to_ngram)
    symbol_dict = vectorize_utils.create_symbol_dict(all_bigrams)
    return symbol_dict

def initialize_label_dict(train_data):
    all_labels = set([lb for lb, _ in train_data])
    label_dict = vectorize_utils.create_symbol_dict(all_labels)
    return label_dict

def xy_generator(corpus, text_to_ngram, symbol_dict, label_dict):
    nx = len(symbol_dict)
    for label, text in corpus:
        ngrams = Counter(text_to_ngram(text)).most_common()
        if len(ngrams):
            y = label_dict[label]
            x = vectorize_utils.generate_vector(nx, ngrams, symbol_dict)
            yield x, y

def load_test_corpus(fname):
    test_data = utils.read_data(fname)
    corpus = [txt for _,txt in test_data]
    return corpus

def predict(trained_params, corpus, text_to_ngram, symbol_dict, label_dict):
    nx = len(symbol_dict)
    rev_label_dict = { v:k for k, v in label_dict.items()}
    for text in corpus:
        ngrams = Counter(text_to_ngram(text)).most_common()
        x = vectorize_utils.generate_vector(nx, ngrams, symbol_dict)
        label_int = ll.predict(x, trained_params)
        label_char = rev_label_dict[label_int]
        yield label_char


def main(text_to_ngram):
    train_data = utils.read_data(config.filename_train)
    if config.loglin.cleanup:
        train_data = [(ln, utils.cleanup(txt, config.cleanup_config)) for ln, txt in train_data]
    symbol_dict = initialize_symbol_dict(train_data, text_to_ngram)
    label_dict = initialize_label_dict(train_data)
    xy_train = list(xy_generator(train_data, text_to_ngram, symbol_dict, label_dict))

    dev_data = utils.read_data(config.filename_dev)
    xy_dev = list(xy_generator(dev_data, text_to_ngram, symbol_dict, label_dict))
    in_dim = min(config.max_count, len(symbol_dict))
    out_dim = len(label_dict)
    print("problem dimensions are: {}".format((in_dim, out_dim)))
    params = ll.create_classifier(in_dim, out_dim)
    trained_params = train_classifier(xy_train, xy_dev, config.loglin.max_iterations, config.loglin.learning_rate, params)
    w, b = trained_params
    k = 7
    print("top {} scores for each language:".format(k))
    for c in range(w.shape[1]):
        top_k = set(np.argsort(w[:,c])[-k:])
        print("language: {} bigrams: {}".format(
            [k for k,v in label_dict.items() if v == c],
            [(k, np.round(w[v,c],2)) for k,v in symbol_dict.items() if v in top_k]
            )
        )  
    if config.loglin.predict_on_test:
        test_corpus = load_test_corpus(config.filename_test)
        print('\n\n\nsaving predictions for test data, size: {}'.format(len(test_corpus)))
        predictions = predict(trained_params, test_corpus, text_to_ngram, symbol_dict, label_dict)
        utils.save_strings_to_file(config.filename_pred, predictions)
    return trained_params



if __name__ == '__main__':
    trained_params = main(utils.text_to_bigrams)


