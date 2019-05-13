import numpy as np
from collections import namedtuple
import argparse

abalone_datum = namedtuple('seashell','sex,length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight'.split(','))

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

class seashell_data_holder:
    @classmethod
    def from_file(cls, x_file, y_file=None):
        self = cls()
        with open(x_file,'rt') as x:
            for line in x:
                self.train_x.append(seashell_data_holder.load_line(line))    
                
        if y_file:
            self.train_y = []
            with open(y_file,'rt') as y:
                for line in y:
                    self.train_y.append(int(line.split('.')[0]))
        self.n_samples = len(self.train_x)
        return self


        
    def __init__(self):
        self.train_x = []
        self.train_y = None
        # here monkey-patching is used to allow flexibility in choosing the converter function
        # in this way, one line of code only (the below line) needs to change to test different converters
        # the static method will still recieve self as first parameter
        self.n_samples = 0 
        self.n_features = 11


    def split(self, counts, shuffle=False):
        """
        return an array of seashell_data_holders, by splitting the existing data, without deep-copying the data
        """
        ty = self.train_y is not None
        if type(counts) ==int:
            counts = [counts]
        if counts[0] != 0:
            counts = [0] + counts
        elif len(counts) == 1:
            raise ValueError("cannot split at 0")
        counts += [self.n_samples]

        ret = [] 
        for (st, sp) in zip(counts[:-1],counts[1:]):
            cur = seashell_data_holder()
            cur.train_x = self.train_x[st:sp]
            if ty:
                cur.train_y = self.train_y[st:sp]
            cur.n_samples = sp-st
            ret.append(cur)
        return ret

    @staticmethod
    def get_datum(datum):
        ret = seashell_data_holder._convert_onehot(datum)
        ret = seashell_data_holder.add_constant(ret)
        return ret

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

    @staticmethod
    def add_constant(vec):
        return np.concatenate([vec,np.ones(1)])

    def data_generator(self, shuffle=True):
        r = np.arange(start=0, stop=self.n_samples,dtype=int)
        if shuffle:
            np.random.shuffle(r)
        for i in r:
            yield self.get_datum(self.train_x[i]), self.train_y[i]


class learn_rate_schedule:
    def __init__(self, alpha=2  ):
        self.eta = 0.1
        self.alpha = alpha
    
    def lr_generator(self):
        while True:
            yield self.eta
            self.eta = self.eta - self.alpha * self.eta**2

class base_classifier:
    def __init__(self, feature_count):
        self.type = None   
        self.nclasses = 3   
        self.w = np.zeros((self.nclasses, feature_count), dtype=np.float)
        self.epochs = 2

    def _score(self, sample_x):
        return np.dot(self.w,sample_x[:,np.newaxis])

    def update_rule(self, sample_x, sample_y, sample_yhat):
        raise NotImplementedError

    def test(self, sample_x):
        return np.argmax(self._score(sample_x))

    def train(self, train_dh, validation_dh):
        for ep in range(self.epochs):
            for sample_x, sample_y in train_dh.data_generator():
                sample_yhat = self.test(sample_x)
                self.update_rule(sample_x, sample_y, sample_yhat)
            good = 0
            for sample_x, sample_y in validation_dh.data_generator(shuffle=False):
                good = good + int(sample_y ==  self.test(sample_x))
            print("epoch: {}, good: {} ({:.1%})".format(ep, good, good/validation_dh.n_samples))
        return good

class pereceptron(base_classifier):
    def __init__(self, feature_count):
        super().__init__(feature_count)
        self.type = 'perceptron'
        self.lr = learn_rate_schedule().lr_generator()
        

    def update_rule(self, sample_x, sample_y, sample_yhat):
        delta = next(self.lr) * sample_x
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


def even_sampling(vec,count):
    """
    returns the entries at (k*i) positions in the sorted vec, where i are consecutive natural numbers and k is len/count
    uses interpolation in case of noninteger index
    """
    svec = sorted(vec)
    x = np.arange(0,1,1/count)
    xp = np.arange(0,1,1/len(svec))
    return np.interp(x,xp,svec)

def main():
    global fprint
    args = parse_args()
    mfp = manage_fprint(args)
    fprint = mfp.get_fprint()

    data = seashell_data_holder.from_file("train_x.txt","train_y.txt")
    validation_set1, validation_set2, train_data = data.split([300,600])
    fiers, goods = select_best_classifier(pereceptron,train_data,validation_set1)
    test_scores = [sum(p.test(x)==y for x,y in validation_set2.data_generator()) for p in fiers]
    print("goods: {}\ntest_goods: {}, corr{}".format(goods, test_scores, np.corrcoef(np.array(goods), np.array(test_scores))))
    
    # print("\n\n\nPassive-Agrressive")
    # pa.train(train_data,validation_data)
    # print("\n\n\n SVM")
    # svm.train(train_data,validation_data)

def select_best_classifier(classifier, train_data, validation_data, attempts=20):
    fiers = []
    goods = np.zeros(attempts, dtype=int)
    for a in range(attempts):
        fiers.append(classifier(train_data.n_features))
        goods[a] = fiers[-1].train(train_data,validation_data)
    return fiers, goods

if __name__ == "__main__":
    main()
