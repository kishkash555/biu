import math
import copy

global d_tree_str

class Data():
    def attributes_options(self):
        attribute_index = 0
        for attribute in self.attributes:
            att_ops = set([line[attribute_index] for line in self.data_lst[1:]])
            self.att_ops_dict[self.attributes[attribute_index]] = att_ops
            attribute_index += 1

    def create_data_dicts(self):
        for line in self.data_lst[1:]:
            n_line = copy.deepcopy(line)
            line_dict = {}
            for att in self.attributes:
                line_dict[att] = n_line.pop(0)
            self.data_dicts.append(line_dict)

    def get_data_from_file(self,file_path):
        with open(file_path, 'r') as f:
            self.data_lst = [line.strip().split('\t') for line in f]
        f.close()

    def __init__(self, data_src, is_file = True):
        if is_file:
            self.get_data_from_file(data_src)
        else:
            self.data_lst = data_src
        self.attributes = self.data_lst[0]
        self.clss_att = self.attributes[-1]
        self.data_dicts = []
        self.create_data_dicts()
        self.att_ops_dict = {}
        self.attributes_options()
        self.clss_ops = list(self.att_ops_dict[self.clss_att])

class Classification_alg():
    # virtual method
    def prediction(self, element):
        None

    def __init__(self, data):
        self.data_lst = data.data_lst
        self.attributes = data.attributes
        self.data_dicts = data.data_dicts
        self.att_ops_dict = data.att_ops_dict
        self.clss_ops = data.clss_ops
        self.clss_att = data.clss_att

class NaiveBayes(Classification_alg):
    p_classes_dic = {}
    c_classes_dic = {}

    def calc_p_clsses(self):
        clss_1_cntr = 0
        for line in self.data_dicts:
            if line[self.clss_att] == self.clss_ops[0]:
                clss_1_cntr += 1
        data_len = len(self.data_dicts)
        self.p_classes_dic[self.clss_ops[0]] = float(clss_1_cntr)/data_len
        self.p_classes_dic[self.clss_ops[1]] = 1 - float(clss_1_cntr)/data_len
        self.c_classes_dic[self.clss_ops[0]] = clss_1_cntr
        self.c_classes_dic[self.clss_ops[1]] = data_len - clss_1_cntr

    def calc_p_att_givven_clss_op(self,att,att_ops, clss_op):
        counter = 0
        for line in self.data_dicts:
            if line[att] == att_ops and line[self.clss_att] == clss_op:
                counter += 1
        return float(counter)/self.c_classes_dic[clss_op]

    def prediction(self, element_dict):
        clss_op_1 = self.clss_ops[0]
        clss_op_2 = self.clss_ops[1]
        p_clss_1 = self.p_classes_dic[clss_op_1]
        p_clss_2 = self.p_classes_dic[clss_op_2]
        for att in element_dict:
            p_clss_1 *= self.calc_p_att_givven_clss_op(att,element_dict[att], clss_op_1)
            p_clss_2 *= self.calc_p_att_givven_clss_op(att,element_dict[att], clss_op_2)
        if p_clss_1 > p_clss_2:
            return clss_op_1
        return clss_op_2


    def __init__(self, data):
        Classification_alg.__init__(self, data)
        self.calc_p_clsses()

class KNN(Classification_alg):
    k = 5

    def calc_h_dst(self, line_index, element_dict):
        h_dst = 0
        for att in self.attributes[:-1]:
            if self.data_dicts[line_index][att] != element_dict[att]:
                h_dst += 1
        return h_dst

    def get_data_knn_index_lst(self, element_dict):
        data_h_dst_dict = {}
        for index in range(len(self.data_dicts)):
            data_h_dst_dict[index] = self.calc_h_dst(index,element_dict)

        # sort data_h_dst_dict
        data_knn_index_lst = [k for k in sorted(data_h_dst_dict, key=data_h_dst_dict.get)]
        # print(data_knn_index_lst)
        return data_knn_index_lst[:self.k]

    def prediction(self, element_dict):
        data_knn_index_lst = self.get_data_knn_index_lst(element_dict)
        knn_clss_lst = []
        for index in data_knn_index_lst:
            knn_clss_lst.append(self.data_dicts[index][self.clss_att])

        clss_1_cntr = 0
        for p in knn_clss_lst:
           if p == self.clss_ops[0]:
               clss_1_cntr += 1
        if clss_1_cntr > self.k/2:
            return self.clss_ops[0]
        return self.clss_ops[1]

    def __init__(self, data):
        Classification_alg.__init__(self, data)

class Decision_Tree(Classification_alg):
    global d_tree_str
    d_tree_str = ''

    def write_dt_to_file(self, file_path):
        d_tree_file = open(file_path, 'w')
        d_tree_file.write(d_tree_str)
        d_tree_file.close()

    def prediction(self, element):
        if self.is_leaf:
            return self.leaf_val
        for child in self.children:
            if child.parent_ops == element[self.bst_attribute]:
                return child.prediction(element)

    def print_tree(self):
        global d_tree_str
        is_first_line = True

        if self.is_leaf:
            d_tree_str += (':' + self.leaf_val)
            return
        tabs = '\n'
        if self.parent != None:
            tabs += self.tabs + '|'
        for child in self.children:
            if self.parent == None and is_first_line == True:
                is_first_line = False
                d_tree_str += (self.bst_attribute + '=' + child.parent_ops)
            else:
                d_tree_str += (tabs + self.bst_attribute + '=' + child.parent_ops)
            child.print_tree()



    def entropy(self, ops_ctr, cls_ctr):
        op1 = float(cls_ctr) / ops_ctr
        op2 = float(ops_ctr - cls_ctr) / ops_ctr
        if 0 < op1 and 0 < op2:
            return -op1 * math.log(op1, 2) - op2 * math.log(op2, 2)
        else:
            return 0

    def gain(self, attribute):
        att_gain = 0
        data_len = len(self.data_dicts)
        for ops in self.att_ops_dict[attribute]:
            ops_ctr = sum(1 for line in self.data_dicts if line[attribute] == ops)
            cls_ctr = sum(1 for line in self.data_dicts if line[attribute] == ops and line[self.clss_att] == self.clss_ops[0])
            p = float(ops_ctr) / data_len
            e = self.entropy(ops_ctr, cls_ctr)
            att_gain += (float(ops_ctr) / data_len) * e

        return att_gain

    def find_best_attribute(self):
        bst_gain = 2.0
        bst_att = ''
        for attribute in self.attributes[:-1]:
            att_gain = self.gain(attribute)
            if att_gain < bst_gain:
                bst_gain = att_gain
                bst_att = attribute
        self.bst_attribute = bst_att

    def get_new_data(self, ops):
        new_data = []
        att_index = self.attributes.index(self.bst_attribute)
        new_data_attributes = copy.deepcopy(self.attributes)
        del new_data_attributes[att_index]
        new_data.append(new_data_attributes)
        for line in self.data_dicts:
            if line[self.bst_attribute] == ops:
                new_line = copy.deepcopy(line)
                new_line.pop(self.bst_attribute)
                new_data.append(list(new_line.values()))

        return new_data

    def __init__(self, data, parent=None, parent_ops=None, tabs=''):
        Classification_alg.__init__(self, data)
        self.parent = parent
        self.tabs = tabs
        self.parent_ops = parent_ops
        self.children = []
        self.is_leaf = False

        if len(self.attributes) > 1:
            self.find_best_attribute()
            for ops in self.att_ops_dict[self.bst_attribute]:
                is_data_from_file = False
                new_data_lst = self.get_new_data(ops)
                new_data = Data(new_data_lst, is_data_from_file)
                self.children.append(Decision_Tree(new_data, self, ops, '\t' + self.tabs))
        # is leaf
        else:
            self.is_leaf = True
            self.leaf_val = self.data_lst[1][0]


def is_all_data_same_class(data):
    data.pop(0)
    class_t = data[0][-1]
    for line in data:
        if line[-1] != class_t:
            return False, ''
    return True, class_t


if __name__ == "__main__":

    train_data = Data("train.txt")
    tst_data = Data("test.txt")

    my_dt = Decision_Tree(train_data)
    my_dt.print_tree()
    my_dt.write_dt_to_file('output_tree_ruth.txt')

    my_knn = KNN(train_data)
    my_nb = NaiveBayes(train_data)

    algo_lst = []
    algo_lst.append(my_dt)
    algo_lst.append(my_knn)
    algo_lst.append(my_nb)

    dt_acc_cntr = 0
    knn_acc_cntr = 0
    nb_acc_cntr = 0

    index = 0
    output_str = 'Num\tDT\tKNN\tnaiveBase\n'

    algo_acc_ctr_dict = {}
    for algo in algo_lst:
        algo_acc_ctr_dict[type(algo)] = 0

    for element in tst_data.data_dicts:
        index += 1
        d = copy.deepcopy(element)
        d.pop(train_data.clss_att)

        element_result = element[train_data.clss_att]
        output_str += (index.__str__() + '.')
        for algo in algo_lst:
            p = algo.prediction(d)
            output_str += ('\t' + p)
            if p == element_result:
                algo_acc_ctr_dict[type(algo)] += 1
        output_str += '\n'

    tst_len = len(tst_data.data_dicts)
    for algo in algo_lst:
        accuracy = round(float(algo_acc_ctr_dict[type(algo)])/tst_len,1)
        output_str += ('\t' + accuracy.__str__())

    output_str += '\n'

    out_file = open('output.txt', 'w')
    out_file.write(output_str)
    out_file.close()
