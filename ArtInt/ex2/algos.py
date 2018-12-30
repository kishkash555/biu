from collections import namedtuple, Counter, defaultdict
from math import log2

def read_input(lines):
    init = False
    cases = []
    for line in lines:
        if not init:
            field_names = line.strip().split('\t')
            init = True
            data_tuple=namedtuple('data_tuple',field_names)
            continue
        values = line.strip().split('\t')
        cases.append(data_tuple(*values))
    return cases

def hamming_distance(v1, v2):
    return sum([0 if x==y else 1 for x, y in zip(v1[:-1], v2[:-1])])


class classifier:
    def fit(self, train_data):
        raise NotImplementedError
    def predict(self, test_data):
        raise NotImplementedError

class knn(classifier):
    def __init__(self, k, distance_func):
        classifier.__init__(self)
        self.train_data = None
        self.k = k
        self.distance_func = distance_func
        
    def fit(self, train_data):
        self.train_data = train_data

    def predict(self, test_data):
        ret = []
        train_data = self.train_data
        distance_func = self.distance_func
        if len(test_data)==0:
            return ret
        applied_labels = []
        train_labels = [train_case[-1] for train_case in train_data]
        for test_case in test_data:
            distances = [distance_func(train_case, test_case) for train_case in train_data]
            k_smallest = argsort(distances)[:self.k]
            applied_labels.append(Counter([train_labels[i] for i in k_smallest]).most_common()[0][0])
        return applied_labels

class decision_tree(classifier):
    min_cases = 5
    def __init__(self):
        classifier.__init__(self)
        self.tree = None
        self.classes = None

    def fit(self,train_data):

        self.classes = sorted(set(map(lambda a: a[-1],train_data)))
        self.data_fields = train_data[0]._fields
        self.decision_variable = self.data_fields[-1]
        tree = {'root': {} }
        self.tree = tree
        self.split(tree['root'], train_data, set([self.data_fields[-1]]))
        
    def split(self, tree, current_train, used_list):
        if len(current_train) < decision_tree.min_cases or len(used_list) == len(self.data_fields):
            for f in self.classes:
                tree[(self.decision_variable, f)] = len(self.__filtered(current_train,{self.decision_variable: f}))
            return
        attr_entropies =  {f: self.__calc_entropy(current_train, f) for f in self.data_fields if f not in used_list}
        m = 100
        for k, v in attr_entropies.items():
            if v < m:
                v = m
                split_on = k
        
        for k in set(t._asdict()[split_on] for t in current_train):
            tree[(split_on, k)] = {}
            self.split(tree[(split_on, k)], self.__filtered(current_train, {split_on: k}), used_list.union([split_on]))
            

    @classmethod
    def __filter(cls, case, qualifier):
        case = case._asdict()
        for k, v in qualifier.items():
            if case[k] != v:
                return False
        return True
    
    @classmethod
    def __filtered(cls, cases, qualifier):
        return [case for case in cases if cls.__filter(case, qualifier)]

    def __calc_entropy(self, cases, attr):
        attr_counter = defaultdict(lambda: [0]*len(self.classes))
        for case in cases:
            attr_counter[case._asdict()[attr]][self.classes.index(case[-1])] += 1
        
        totals = []
        subtree_entr = []
        for distrib in attr_counter.values():
            totals.append(sum(distrib))
            subtree_entr.append(entropy_from_distrib(distrib))
        attr_entropy = dot(subtree_entr,totals)/sum(totals)
        return attr_entropy
    
def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)


def entropy_from_distrib(p):
    s = sum(p)
    return -sum(map(lambda x: log2(x/s)*x/s if x > 0 else 0, p))

def dot(p, e):
    return sum(map(lambda a: a[0]*a[1], zip(p,e)))

