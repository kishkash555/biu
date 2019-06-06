import argparse
import numpy as np
import pickle
from data_iterator import data_iterator

def parse_args():
    parser = argparse.ArgumentParser(description="Predict on test data using a model loaded from a file")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument('--data-x', type=str, required=True)
    parser.add_argument('--data-y', type=str, required=False, help='specify this argument if you wish to check the accuracy on a tagged dataset')
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    return args

def load_model(args):
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    return model

def load_unknown_type(fname):
    loaded = True
    try:
        ret = np.load(fname)
    except OSError:
        loaded=False
    if not loaded:
        ret = np.loadtxt(fname)
    return ret

def load_test_data(args):    
    test_arr = load_unknown_type(args.data_x)
    return data_iterator(test_arr, None, args.batch_size, False)

def load_validation_data(args):
    valid_x_arr = load_unknown_type(args.data_x)
    valid_y_arr = load_unknown_type(args.data_y)
    return data_iterator(valid_x_arr, valid_y_arr, args.batch_size, False)

def print_predictions(model, test_data):
    for test_x in iter(test_data):  
        y_hats = np.argmax(model.forward(test_x), axis=1)
        #print(np.array2string(y_hats,max_line_width=1e6))
        print('\n'.join(np.array2string(y_hats,max_line_width=1e6).strip('[]').split(' ')), flush=True)

def print_accuracy(model, valid_data):
    val_good = val_cases = 0
    for x, y in iter(valid_data):
        y_hats = np.argmax(model.forward(x), axis=1)
        val_good += (y_hats==y).sum()
        val_cases += len(y)
    report_validation = " acc {:.1%}, ".format(val_good/ val_cases)
    print(report_validation, flush=True)

def main():
    args = parse_args()
    print(args)
    model = load_model(args)
    if args.data_y is not None:
        valid_data = load_validation_data(args)
        print_accuracy(model, valid_data)
        return
    
    test_data = load_test_data(args)
    print_predictions(model, test_data)

if __name__ == "__main__":
    main()