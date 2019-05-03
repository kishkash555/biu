import numpy as np
import librosa
from os import path
import glob

TEST_DIR = "test_files"
TRAIN_DIR = "train_files"
FILES_GLOB = "*.wav"

def main():
    train_files = list(glob.glob(os.path.join(TRAIN_DIR,FILES_GLOB)))
    test_files = list(glob.glob(os.path.join(TEST_DIR,FILES_GLOB)))

    feature_extractor


class feature_extractor:
    def __init__(self):
        pass
    def extract(audio_samples,sr):
        mfcc = librosa.feature.mfcc(y=audio_samples, sr = sr)
        return mfcc

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

    def predict(self, test_sample):
        dist = self.distance_metric(self.train_x, test_sample)
        min_dist_sample = np.argmin(dist)
        return train_y(min_dist_sample)

def dynamic_time_warping(vec1, vec2):
    """
    DTW algorithm computation
    expecting a numpy 1d vector on each side
    """

    v1 = vec1.squeeze()
    v2 = vec2.squeeze()
    t1, t2 = v1.shape[0], v2.shape[0]

    ret = np.zeros((t1, t2), dtype=np.float)

    # compute the DTW distance matrix
    # compute the distance along first row
    ret[:,0] = np.cumsum(np.abs(v1-v2[0]))
    # compute the distance along first column
    ret[0,:] = np.cumsum(np.abs(v1[0] - v2))

    for i in range(1,t1):
        for j in range(1,t2):
            ret[i,j] = np.abs(v1[i]-v2[j]) + min(ret[i-1,j-1], ret[i,j-1], ret[i-1,j])
            print("({},{}):{}".format(v1[i],v2[j],ret[i,j]))
        
    return ret[t1,t2]

if __name__ == "__main__":
    vec1 = np.array([1,6,2,3,0,9,4,3,6,3], dtype=np.float)
    vec2 = np.array([1,3,4,9,8,2,1,5,7,3], dtype=np.float)
    ret = dynamic_time_warping(vec1, vec2)
    1

    # y, sr = librosa.load(f_path, sr=None)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr)
