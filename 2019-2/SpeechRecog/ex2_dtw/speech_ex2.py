import numpy as np
import librosa
from os import path
import glob
from collections import defaultdict
import argparse

TEST_DIR = "test_files"
TRAIN_DIR = "train_files"
FILES_GLOB = "*.wav"

N_FEATURES = 20
LABELS_TO_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
    }
    
euclidean_dist = lambda v1, v2: np.linalg.norm(v2-v1)**2

class naive_feature_extractor:
    @staticmethod
    def extract(audio_samples,sr):
        mfcc = librosa.feature.mfcc(y=audio_samples, sr = sr, n_mfcc=N_FEATURES)
        return mfcc

class truncate_feature_extractor:
    @staticmethod
    def extract(audio_samples,sr):
        mfcc = librosa.feature.mfcc(y=audio_samples, sr = sr, n_mfcc=N_FEATURES) # 20x32
        mfcc = mfcc[:-1,:]        
        return mfcc
        
extract = naive_feature_extractor.extract

def load(fname):
    return librosa.load(fname, sr=None)

def main(is_normalize_inputs=False, is_normalize_features=False):
    train_files = list(glob.glob(path.join(TRAIN_DIR,'*',FILES_GLOB)))
    test_files = sorted(glob.glob(path.join(TEST_DIR,FILES_GLOB)))
    
    train_x = []
    train_y = []

    for trn in train_files:
        label = path.split(path.split(trn)[0])[1]
        x_trn, sr_train = load(trn)
        if is_normalize_inputs:
            x_trn = normalize_vec(x_trn)
        feature_mat = extract(x_trn,sr_train).T
        if is_normalize_features:
            feature_mat = normalize_columns(feature_mat)
        train_x.append(feature_mat)
        train_y.append(LABELS_TO_NUMBERS[label])
        


    clsfier1 = nn1(lambda v1, v2: dynamic_time_warping(v1, v2, euclidean_dist)).fit(train_x, train_y)
    clsfier2 = nn1(sum_of_euclid_distances).fit(train_x, train_y)
    
    for tst in test_files:
        tst_fname = path.basename(tst)
        x_test, sr_test = load(tst)
        
        if is_normalize_inputs:
            x_test = normalize_vec(x_test)
        
        tst_features = extract(x_test, sr_test).T
        if is_normalize_features:
            tst_features = normalize_columns(tst_features)

        cls1 = clsfier1.predict(tst_features)
        cls2 = clsfier2.predict(tst_features)

        fprint("{} - {} - {}".format(tst_fname, cls2, cls1))
        

def normalize_vec(v):
    return (v-v.mean()) / v.std()

def normalize_columns(m):
    return (m-np.average(m,axis=0))/np.std(m,axis=0)

class nn1:
    def __init__(self, distance_metric):
        """
        distance metric: a function which accepts a 2d np-array and a 1d np array.
            returns a 1d np array with all the distances
        """
        self.train_x = None
        self.train_y = None
        self.distance_metric = distance_metric

    def fit(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        return self

    def predict(self, test_sample):
        dist = np.array([self.distance_metric(x, test_sample) for x in self.train_x])
        min_dist_sample = np.argmin(dist)
        return self.train_y[min_dist_sample]

    def predict_full(self, test_sample):
        dist = np.array([self.distance_metric(x, test_sample) for x in self.train_x])
        ret = sorted(zip(self.train_y,dist))
        return ret

    def save_data(self,fname):
        save_arr = np.array(
            sorted(
                [[y]+x for x,y in zip(self.train_x, self.train_y)],
                key=lambda r: r[0]
                ),
            dtype=np.float)
        np.savetxt(fname, save_arr,fmt='%.2f',delimiter=',')

def sum_of_euclid_distances(v1,v2):
    return sum(np.linalg.norm(v1-v2,axis=1))       

def dynamic_time_warping(vec1, vec2, distance_metric):
    """
    DTW algorithm computation
    expecting a numpy 2d array on each side
    each row contains the coordinate at a point
    """
    dist = distance_metric
    v1, v2 = vec1, vec2
    t1, t2 = v1.shape[0], v2.shape[0]

    ret = np.zeros((t1, t2), dtype=np.float)

    # compute the DTW distance matrix
    # compute the distance along first row
    for i in range(t1):
        ret[i,0] = dist(v1[i,:],v2[0,:])
    ret[:,0] = np.cumsum(ret[:,0])
    # compute the distance along first column
    for i in range(t2):
        ret[0,i] = dist(v1[0,:],v2[i,:])
    ret[0,:] = np.cumsum(ret[0,:])

    for i in range(1,t1):
        for j in range(1,t2):
            ret[i,j] = dist(v1[i,:],v2[j,:]) + min(ret[i-1,j-1], ret[i,j-1], ret[i-1,j])
            
    return ret[-1,-1]

def test_dtw():
    vec1 = np.array([[1,6,2,3,0,9,4,3,6,3]], dtype=np.float).T
    vec2 = np.array([[1,3,4,9,8,2,1,5,7,3]], dtype=np.float).T
    dist = lambda v1, v2: np.abs(v1[0]-v2[0])
    ret = dynamic_time_warping(vec1, vec2, dist)
    assert(ret==15)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', action='store', nargs='?')
    args = parser.parse_args()
    return args

class manage_fprint:
    def __init__(self, args):
        self.f = None
        if args.output_file:
            self.f = open(args.output_file,'wt',buffering=1)

    def get_fprint(self):
        def fprint(s):
            print(s)
            print(s, file=self.f)
        if self.f:
            return fprint
        return print

    def close(self):
        self.f.close()


if __name__ == "__main__":
    global fprint
    args = parse_args()
    mfp = manage_fprint(args)
    fprint = mfp.get_fprint()
    
    main(is_normalize_inputs=True, is_normalize_features=True)
    mfp.close()



