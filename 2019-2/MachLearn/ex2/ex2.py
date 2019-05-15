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
        self._to_array()
        return self


        
    def __init__(self):
        self.train_x = []
        self.train_y = None
        # here monkey-patching is used to allow flexibility in choosing the converter function
        # in this way, one line of code only (the below line) needs to change to test different converters
        # the static method will still recieve self as first parameter
        self.n_samples = 0 
        self.n_features = None
        self.array_x = None

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
            cur._to_array()
            ret.append(cur)
        return ret

    def _to_array(self):
        self.array_x = np.vstack([
            seashell_data_holder._convert_onehot(
                seashell_data_holder.approx_mean(x)
                ) for x in self.train_x])
        self.add_constant()
        self.n_features = self.array_x.shape[1]


    @staticmethod
    def load_line(line):
        sp_line = line.split(',')
        ret = [sp_line[0]] + [float(d) for d in sp_line[1:]]
        return abalone_datum(*ret)
    
    @staticmethod
    def approx_mean(datum):
        return abalone_datum(
            datum.sex,
            datum.length - 0.5,
            datum.diameter - 0.4,
            datum.height - 0.14,
            datum.whole_weight - 0.8,
            datum.shucked_weight - 0.34,
            datum.viscera_weight - 0.17,
            datum.shell_weight - 0.23
            )

    @staticmethod
    def _convert_onehot(datum):
        a1 = [0.,0.,0.]
        d = {"M": 0, "F": 1, "I": 2}
        a1[d[datum.sex]]=1.
        a2 = list(datum)
        ret = np.array(a1+a2[1:], dtype=float)
        return ret

    def add_constant(self):
        self.array_x = np.hstack([self.array_x, np.ones((self.array_x.shape[0],1))])

    
    def add_2nd_degree_full(self, fields):
        lf = len(fields)
        n_added = int(lf*(lf+1)/2)
        ax = self.array_x
        added_fields = np.zeros((ax.shape[0],n_added),dtype=float)
        curr = 0
        for i in range(lf):
            for j in range(i,lf):
                added_fields[:,curr] = ax[:,fields[i]]*ax[:,fields[j]]
                curr += 1
        self.array_x = np.concatenate([ax,added_fields])

    def add_2nd_degree(self, fields):
        lf = len(fields)
        n_added = lf
        ax = self.array_x
        added_fields = np.zeros((ax.shape[0],n_added),dtype=float)
        curr = 0
        for i in range(lf):
            added_fields[:,curr] = ax[:,fields[i]]*ax[:,fields[i]]
            curr += 1
        self.array_x = np.concatenate([ax,added_fields])

    def get_train_x_as_array(self):
        return np.vstack([x[0] for x in self.data_generator(False)])

    def digitize(self, digitization_datum):
        ax = self.array_x
        for i, d_array in enumerate(digitization_datum):
            if not (d_array is None):
                self.array_x[:,i] = np.digitize(ax[:,i], d_array)
        
    def get_digitization_func(self, n_points):
        ax = self.array_x
        feature_list = [np.sort(ax[:,i]) for i in range(ax.shape[1])]
        xp = np.arange(ax.shape[0])
        x = np.linspace(0,ax.shape[0],n_points)
        funcs = [np.interp(x, xp, f) for f in feature_list]
        return funcs

    def data_generator(self, shuffle=True):
        r = np.arange(start=0, stop=self.n_samples, dtype=int)
        if shuffle:
            np.random.shuffle(r)
        for i in r:
            yield self.array_x[i,:], self.train_y[i]


class learn_rate_schedule:
    def __init__(self, alpha=2  ):
        self.eta = 0.1
        self.alpha = alpha
        self.lr_generator = self.inverse_time_decay
    
    def inverse_time_decay(self):
        while True:
            yield self.eta
            self.eta = self.eta - self.alpha * self.eta**2

    def exponential_decay(self):
        if self.alpha >= 1.:
            raise ValueError("alpha ({}) must be <1".format(self.alpha))
        while True:
            yield self.eta
            self.eta = self.alpha * self.eta

class base_classifier:
    def __init__(self, feature_count):
        self.type = None   
        self.nclasses = 3   
        self.w = np.zeros((self.nclasses, feature_count), dtype=np.float)
        self.epochs = 1

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
            #fprint("epoch: {}, good: {} ({:.1%})".format(ep, good, good/validation_dh.n_samples))
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
        tau = loss / (2*np.dot(sample_x, sample_x))
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
        self.type = 'svm'

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
    validation_set1, validation_set2, train_data = data.split([300, 600])

    fiers = select_best_classifier(pereceptron, train_data, validation_set1, return_all=True)
    test_scores = np.array([sum(p.test(x)==y for x,y in validation_set2.data_generator()) for p in fiers])
    fprint("perceptron test_scores raw features:\n{}\n{} +/- {}".format(
        np.array2string(test_scores,separator=', '), 
        np.mean(test_scores), np.std(test_scores)
        ))

    digitaztion_datum = train_data.get_digitization_func(50)
    digitaztion_datum[0:3] = None, None, None # do not digitize sex
    digitaztion_datum[10] = None # do not digitize constant
    train_data.digitize(digitaztion_datum)
    validation_set1.digitize(digitaztion_datum)
    validation_set2.digitize(digitaztion_datum)
    fiers = select_best_classifier(pereceptron, train_data, validation_set1, return_all=True)
    test_scores = np.array([sum(p.test(x)==y for x,y in validation_set2.data_generator()) for p in fiers])
    fprint("percptron test_scores with digitze:\n{}\n{} +/- {}".format(
        np.array2string(test_scores,separator=', '), 
        np.mean(test_scores), 
        np.std(test_scores)
        ))
    

def select_best_classifier(classifier, train_data, validation_data, attempts=20,return_all=False):
    fiers = []
    goods = np.zeros(attempts, dtype=int)
    for a in range(attempts):
        fiers.append(classifier(train_data.n_features))
        goods[a] = fiers[-1].train(train_data,validation_data)
    if return_all:
        return fiers
    return fiers[np.argmax(goods)]

if __name__ == "__main__":
    main()
