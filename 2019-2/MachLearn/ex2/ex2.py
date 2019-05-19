import argparse
from collections import namedtuple
import numpy as np
from os import path

abalone_datum = namedtuple('seashell','sex,length,diameter,height,whole_weight,shucked_weight,viscera_weight,shell_weight'.split(','))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_x', action='store')
    parser.add_argument('train_y', action='store')
    parser.add_argument('test_x', action='store')

    args = parser.parse_args()
    return args

def parse_args_debug():
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
            self.train_y = np.array(np.round(np.loadtxt(y_file)),dtype=int)
        
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

    def split(self, counts, k_fold=False, shuffle=False):
        """
        return an array of seashell_data_holders, by splitting the existing data, without deep-copying the data
        """
        ty = self.train_y is not None
        if shuffle:
            shuffle_ind = np.arange(self.n_samples)
            np.random.shuffle(shuffle_ind)
            self.train_x = [self.train_x[i] for i in shuffle_ind]
            self.array_x = self.array_x[shuffle_ind,:] 
            if ty:
                self.train_y = self.train_y[shuffle_ind] 
            
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
            if k_fold:
                cur2 = seashell_data_holder()
                cur2.train_x = self.train_x[:st]+ self.train_x[sp:]
                if ty:
                    cur2.train_y = np.concatenate([self.train_y[:st], self.train_y[sp:]])
                cur2.n_samples = self.n_samples - (sp-st)
                cur2._to_array()
                ret.append((cur,cur2))
            else:
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
        a2 = list(datum) # convert tuple to list
        ret = np.array(a1+a2[1:], dtype=float)
        return ret

    def add_constant(self):
        self.array_x = np.hstack([self.array_x, np.ones((self.array_x.shape[0],1))])

    
    def data_generator(self, shuffle=True):
        r = np.arange(start=0, stop=self.n_samples, dtype=int)
        if shuffle:
            np.random.shuffle(r)
        for i in r:
            yield self.array_x[i,:], None if self.train_y is None else self.train_y[i]


class learn_rate_schedule:
    def __init__(self, lr_type, **params):
        self.params = params
        if lr_type == 'exponential':
            self.lr_generator = self.exponential_decay()
        elif lr_type == 'inverse_time':
            self.lr_generator = self.inverse_time_decay()
        elif lr_type == 'constant':
            self.lr_generator = self.constant_lr()
    
    def inverse_time_decay(self):
        step = 0
        alpha = self.params['alpha']
        eta = self.params['eta']
        while True:
            step += 1
            yield eta/(1 + alpha* step)

    def exponential_decay(self):
        alpha = self.params['alpha']
        eta = self.params['eta']
        if alpha >= 1.:
            raise ValueError("alpha ({}) must be <1".format(alpha))
        while True:
            yield eta
            eta *= alpha 
    
    def constant_lr(self):
        eta = self.params['eta']
        while True:
            yield eta



class base_classifier:
    def __init__(self, feature_count, debug):
        self.type = None   
        self.nclasses = 3   
        self.w = np.zeros((self.nclasses, feature_count), dtype=np.float)
        self.epochs = 20
        self.debug = bool(debug)

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
            if self.debug:
                for sample_x, sample_y in validation_dh.data_generator(shuffle=False):
                    good = good + int(sample_y ==  self.test(sample_x))
                    fprint("{} ep {} accuracy: {} ({:.1%})".format(self.type, ep, good, good/validation_dh.n_samples))
            self.next_epoch()
        return good
    
    def next_epoch(self):
        raise NotImplementedError

class pereceptron(base_classifier):
    def __init__(self, feature_count, debug=False):
        super().__init__(feature_count, debug)
        self.epochs = 15
        self.type = 'perceptron'
        self.lr = learn_rate_schedule('exponential', eta = 0.1, alpha = 0.5).lr_generator
        self.next_epoch()        

    def update_rule(self, sample_x, sample_y, sample_yhat):
        delta = self.eta * sample_x
        self.w[sample_y, :] += delta
        self.w[sample_yhat, :] -= delta

    def next_epoch(self):
        self.eta = next(self.lr)


class passive_agressive(base_classifier):
    def __init__(self, feature_count, debug=False):
        super().__init__(feature_count, debug)
        self.type = 'pa'
        self.max_tau_rate = learn_rate_schedule('exponential', eta = 1, alpha = 0.5).lr_generator
        self.next_epoch()
    
    def update_rule(self, sample_x, sample_y, sample_yhat):
        loss = self.hinge_loss(sample_x, sample_y)
        tau = loss / (2*np.dot(sample_x, sample_x))
        tau = min(tau, self.max_tau)
        delta = tau * sample_x
        self.w[sample_y,:] += delta
        self.w[sample_yhat, :] -= delta

    def hinge_loss(self, sample_x, sample_y):
        score = self._score(sample_x) 
        true_class_score = float(score[sample_y])
        score[sample_y] = -np.inf
        highest = np.amax(score)
        return max(0, 1- (true_class_score - highest))        
    
    def next_epoch(self):
        self.max_tau = next(self.max_tau_rate)
 

class support_vector_machine(base_classifier):
    def __init__(self, feature_count, debug=False):
        super().__init__(feature_count, debug)
        self.lada = 0.1 # "lambda" is reserved
        self.type = 'svm'
        self.lr = learn_rate_schedule('exponential', eta = 0.1, alpha = 0.5).lr_generator
        self.next_epoch()

    def update_rule(self, sample_x, sample_y, sample_yhat):
        delta = self.eta * sample_x
        decay = 1- self.lada * self.eta
        self.w = decay * self.w
        self.w[sample_y,:] +=  delta
        self.w[sample_yhat, :] -= delta

    def next_epoch(self):
        self.eta = next(self.lr)

def main_debug():
    global fprint
    args = parse_args_debug()
    mfp = manage_fprint(args)
    fprint = mfp.get_fprint()

    data = seashell_data_holder.from_file("train_x.txt","train_y.txt")
    kfold_data = data.split(list(range(0,data.n_samples,600)),k_fold=True, shuffle=True)

    nf = data.n_features
    for validation, train_data in kfold_data:
        fiers = [
            pereceptron(nf, False),
            support_vector_machine(nf, False),
            passive_agressive(nf, False)
        ]

        test_scores = []
        for p in fiers:
            p.train(train_data, validation)
            test_scores.append(sum(p.test(x)==y for x,y in validation.data_generator()))
        report = ["({}, {}, {:.1%})".format(f.type, score, score/validation.n_samples) for f, score in zip(fiers, test_scores)]
        fprint("test_score: {}".format(report))



def main():
    global fprint
    args = parse_args()
    # args.output_file = None

    # mfp = manage_fprint(args)
    fprint = print

    input_fnames = [args.train_x, args.train_y, args.test_x]

    all_valid = [path.isfile(f) for f in input_fnames]
    if not all(all_valid):
        print("The following paths did not resove to a valid file name: {}".format(", ".join(f for f,b in zip(input_fnames, all_valid) if b is False)))
        raise ValueError("File(s) not found")

    train_data = seashell_data_holder.from_file(args.train_x, args.train_y)
    test_data = seashell_data_holder.from_file(args.test_x)
    #validation_set, train_data = data.split(300)

    fiers = [
        pereceptron(train_data.n_features, False),
        support_vector_machine(train_data.n_features, False),
        passive_agressive(train_data.n_features, False)
    ]
    for p in fiers:
        p.train(train_data,None)
    for case in test_data.data_generator(shuffle=False):
        fprint(", ".join(["{}: {}".format(f.type, f.test(case[0])) for f in fiers]))



def select_best_classifier(classifier, train_data, validation_data, attempts=30,return_all=False):
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
