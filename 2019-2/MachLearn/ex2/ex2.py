import numpy as np
from collections import namedtuple

abalone_datum = namedtuple('seashell','sex,length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight'.split(','))

class seashell_data_holder:
    def __init__(self, x_file, y_file=None):
        self.train_x = []
        
        with open(x_file,'rt') as x:
            for line in x:
                self.train_x.append(seashell_data_holder.load_line(line))    
        if y_file:
            self.train_y = []
            with open(y_file,'rt') as y:
                for line in y:
                    self.train_y.append(int(line.split('.')[0]))
        else:
            self.train_y=None
        
        # here monkey-patching is used to allow flexibility in choosing the converter function
        # in this way, one line of code only (the below line) needs to change to test different converters
        # the static method will still recieve self as first parameter
        self.get_datum = seashell_data_holder._convert_onehot
        self.n_samples = len(self.train_x)
        self.n_features = 10


    @staticmethod
    def load_line(line):
        sp_line = line.split(',')
        ret = [sp_line[0]] + [float(d) for d in sp_line[1:]]
        return abalone_datum(*ret)
    
    @staticmethod
    def _convert_onehot(datum):
        a1 = [0.,0.,0.]
        d = {"M": 0, "F": 1, "I": 2}
        a1[d[datum.sex]]=1.
        a2 = list(datum)
        ret = np.array(a1+a2[1:], dtype=float)
        return ret

    def data_generator(self, shuffle=True):
        r = np.arange(start=0, stop=self.n_samples,dtype=int)
        if shuffle:
            np.random.shuffle(r)
        for i in r:
            yield self.get_datum(self.train_x[i]), self.train_y[i]



# class threeway_predictor:
#     def __init__(self, class0, class1, class2):
#         self.classifiers = [class0, class1, class2]

#     def predict(self, sample_x):
#         votes = np.array([
#             [[1,0,0],[0,1,0],[0,0,1]],
#             [[0,1,1],[1,0,1],[1,1,0]],
#         ], dtype = int)
#         y_total = np.zeros(3, dtype=int)
#         for i,cl in enumerate(self.classifiers):
#             y = 0 if cl.predict(sample_x) < 0 else 1
#             y_total += votes[y, i, :]
#         return np.argmax(y_total)

class base_classifier:
    def __init__(self, feature_count):
        self.type = None   
        self.nclasses = 3   
        self.w = np.zeros((self.nclasses, feature_count), dtype=np.float)
        self.epochs = 20

    def _score(self, sample_x):
        return np.dot(self.w,sample_x[:,np.newaxis])

    def update_rule(self, sample_x, sample_y, sample_yhat):
        raise NotImplementedError

    def test(self, sample_x):
        return np.argmax(self._score(sample_x))

    def train(self, dh):
        for ep in range(self.epochs):
            for sample_x, sample_y in dh.data_generator():
                sample_yhat = self.test(sample_x)
                self.update_rule(sample_x, sample_y, sample_yhat)
            good = 0
            for sample_x, sample_y in dh.data_generator(shuffle=False):
                good = good + int(sample_y ==  self.test(sample_x))
            print("epoch: {}, good: {} ({:.1%})".format(ep, good, good/dh.n_samples))


class pereceptron(base_classifier):
    def __init__(self, feature_count):
        super().__init__(feature_count)
        self.type = 'perceptron'
        self.eta = 0.01

    def update_rule(self, sample_x, sample_y, sample_yhat):
        delta = self.eta * sample_x
        self.w[sample_y, :] += delta
        self.w[sample_yhat, : ] -= delta


class passive_agressive(base_classifier):
    def __init__(self, feature_count):
        super().__init__(feature_count)
        self.tau = 0.01
        self.type = 'pa'
    
    def update_rule(self, sample_x, sample_y, sample_yhat):
        loss = self.hinge_loss(sample_x, sample_y)
        tau = loss / np.dot(sample_x, sample_x)
        delta = tau * sample_x
        self.w[sample_y,:] += delta
        self.w[sample_yhat, :] -= delta

    def hinge_loss(self, sample_x, sample_y):
        score = self._score(sample_x) 
        true_class_score = float(score[sample_y])
        score[sample_y] = -np.inf
        highest = np.amax(score)
        return max(0, 1- (true_class_score - highest))        
 
class support_vector_machine(base_classifier):
    def __init__(self, feature_count):
         super().__init__(feature_count)
         self.eta = 0.01
         self.lada = 0.001
    
    def update_rule(self, sample_x, sample_y, sample_yhat):
        delta = self.eta * sample_x
        decay = 1- self.lada * self.eta
        self.w = decay * self.w
        self.w[sample_y,:] +=  delta
        self.w[sample_yhat, :] -= delta


if __name__ == "__main__":
    train_data = seashell_data_holder("train_x.txt","train_y.txt")
    pcp = pereceptron(train_data.n_features)
    pa = passive_agressive(train_data.n_features)
    svm = support_vector_machine(train_data.n_features)

    # print("Perceptron")
    # pcp.train(train_data)
    print("\n\n\nPassive-Agrressive")
    pa.train(train_data)
    print("\n\n\n SVM")
    svm.train(train_data)