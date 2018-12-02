from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
import sys
import pickle
import config


def load_features(feature_file):
    return load_svmlight_file(feature_file)

def train_model(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

def save_model(model_file, model):
    with open(model_file,'wb') as mf:
        pickle.dump(model, mf)


if __name__ == "__main__":
    if len(sys.argv)==1:
        print("Using default file names")
        feature_file = config.defaultFiles.memm_feature_vec
        model_file = config.defaultFiles.memm_model_file
    elif len(sys.argv) != 3 :
        print(f"usage: {sys.argv[0]} feature_file model_file\nexiting.")
        exit()
    else:
        feature_file = sys.argv[1]
        model_file = sys.argv[2]
    print(f"loading from feature file {feature_file}:")
    train_x, train_y = load_features(feature_file)
    print("done. training...")
    model = train_model(train_x, train_y)
    print(f"done. saving...")
    save_model(model_file, model)
    print("completed successfully")
