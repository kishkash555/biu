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
        self.positive_class = None

    def fit(self,train_data):
        self.classes = sorted(set(map(lambda a: a[-1],train_data)))
        self.positive_class = find_positive_class(self.classes)
        self.data_fields = train_data[0]._fields
        self.decision_variable = self.data_fields[-1]
        tree = {'root': {} }
        self.tree = tree
        self.split(tree['root'], train_data, set([self.data_fields[-1]]))
    
    def predict(self, test_data):
        predictions = []
        for test_case in test_data:
            tree = self.tree['root']
            while True: # loop to go down tree
                next_attr = next(iter(tree.keys()))[0]
                if next_attr == self.decision_variable:
                    chosen = self.get_class_from_leaves(tree) 
                    predictions.append(chosen)
                    break
                else:
                    try:
                        tree = tree[(next_attr,test_case._asdict()[next_attr])]
                    except KeyError:
                        self.get_class_from_leaves(self.get_scores_from_subtree(tree))
        return predictions


    def split(self, tree, current_train, used_list):
        if len(current_train) < decision_tree.min_cases or len(used_list) == len(self.data_fields):
            for f in self.classes:
                tree[(self.decision_variable, f)] = len(self.__filtered(current_train,{self.decision_variable: f}))
            return
        attr_entropies =  {f: self.__calc_entropy(current_train, f) for f in self.data_fields if f not in used_list}
        m = 100
        for k, v in attr_entropies.items():
            if v < m:
                m = v
                split_on = k
        
        for k in set(t._asdict()[split_on] for t in current_train):
            tree[(split_on, k)] = {}
            self.split(tree[(split_on, k)], self.__filtered(current_train, {split_on: k}), used_list.union([split_on]))

    def create_tree_structure_report(self):
        tree = self.tree['root']
        queue = [[k] for k in sorted(tree.keys(), reverse=True)]
        while len(queue):
            curr_key = queue.pop()
            depth = len(curr_key) - 1 
            prefix = '\t'*depth + '|' if depth > 0 else ''
            curr_tree = tree
            for k in curr_key:
                curr_tree = curr_tree[k]
            predicted_classes = self.get_classes_from_subtree(curr_tree)
            if len(predicted_classes)==1:
                suffix = ':' + next(iter(predicted_classes))
                go_down = False
            else:
                suffix = ''
                go_down = True
            line = prefix + "=".join(curr_key[-1]) + suffix 
            yield line
            if go_down:
                queue += [curr_key+[k] for k in sorted(curr_tree.keys(), reverse=True)]
            
    def get_class_from_leaves(self, tree):
        max_v = 0
        for k, v in tree.items():
            if v > max_v or v==max_v and k[1] == self.positive_class:
                predicted_class = k[1]
                max_v = v
        return predicted_class

    def get_classes_from_subtree(self, tree):
        get_first_key= lambda d: next(iter(d.keys()))
    
        if get_first_key(tree)[0] == self.decision_variable:
            return set([self.get_class_from_leaves(tree)])
        predicted_classes = set()
        for subtree in tree.values():
            next_attr_name = get_first_key(subtree)[0]
            if next_attr_name == self.decision_variable: 
                predicted_classes.add(self.get_class_from_leaves(subtree))
            else:
                predicted_classes.update(self.get_classes_from_subtree(subtree))
        return predicted_classes
    
    def get_scores_from_subtree(self, tree):
        get_first_key= lambda d: next(iter(d.keys()))

        if get_first_key(tree)[0] == self.decision_variable:
            return Counter(tree)
        if len(tree)==0 or type(tree)!=dict:
            return Counter()
        print(tree)
        return sum([self.get_scores_from_subtree(t) for t in tree.values()],Counter())

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

def find_positive_class(classes):
    for cl in classes:
            if cl.lower() in {'y','yes','t','1','true'}:
                return cl 

class naive_bayes(classifier):
    def __init__(self):
        classifier.__init__(self)

    def fit(self, train_data):
        self.classes = sorted(set(map(lambda a: a[-1],train_data)))
        self.data_fields = train_data[0]._fields
        self.decision_variable = self.data_fields[-1]
        self.positive_class = find_positive_class(self.classes)
       

        class_counts = Counter()
        feature_class_counts=defaultdict(lambda: defaultdict(Counter))
        for case in train_data:
            case_class = case[-1]
            for feature_name, feature_value in zip(self.data_fields[:-1],case[:-1]):
                feature_class_counts[case_class][feature_name][feature_value] += 1
            class_counts[case_class] += 1
        
        self.class_counts = class_counts
        self.feature_class_counts = feature_class_counts
        
    def predict(self,test_data):
        applied_labels = []
        
        fcc = self.feature_class_counts
        for test_case in test_data:
            probs=[]
            for pclass in self.classes:
                p = -log2(self.class_counts[pclass]/ sum(self.class_counts.values()))
                for feature_name, feature_value in zip(self.data_fields[:-1],test_case[:-1]):
                    prob_fraction = fcc[pclass][feature_name][feature_value] , self.class_counts[pclass]
                    k = len(fcc[pclass][feature_name])
                    p -= log2((prob_fraction[0]+1)/(prob_fraction[1]+k))       
                probs.append(p)
            srt = argsort(probs)
            if probs[srt[0]] == probs[srt[1]]:
                applied_labels.append(self.positive_class)
            else:
                applied_labels.append(self.classes[srt[0]])
        return applied_labels



def argsort(seq):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    return sorted(range(len(seq)), key=seq.__getitem__)


def entropy_from_distrib(p):
    s = sum(p)
    return -sum(map(lambda x: log2(x/s)*x/s if x > 0 else 0, p))

def dot(p, e):
    return sum(map(lambda a: a[0]*a[1], zip(p,e)))

