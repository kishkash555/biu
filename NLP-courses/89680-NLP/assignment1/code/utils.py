import pickle

def load_model(input_file):
    with open(input_file,'rb') as i:
        model = pickle.load(i)
    return model