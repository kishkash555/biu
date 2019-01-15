import _dynet as dy
import network as net
import numpy as np

DIMR_DEPTH = 0
ATTEND_DEPTH = 2
COMPARE_DEPTH = 2
AGG_DEPTH = 2

class decomposable(net.network):
    def __init__(self, word_embeddings, embedding_dim=None, hidden_dim=None, classes_dim=None, pc=None, trained_matrices=None):
        embedding_dim = embedding_dim or 300
        hidden_dim = hidden_dim or 200
        classes_dim = classes_dim or 3

        self.embeddings = word_embeddings
        self.embedding_dim, self.hidden_dim, self.classes_dim = embedding_dim, hidden_dim, classes_dim

            
        if pc and trained_matrices:
            print "loading pretrained inputs"
            self.pc=pc
            first_attend_index = 2 + DIMR_DEPTH * 2 # the *2 accounts for the w and b in each layer
            first_compare_index = first_attend_index + ATTEND_DEPTH * 2 
            first_agg_index = first_compare_index + COMPARE_DEPTH * 2

            print "dimension reducer", range(1, first_attend_index)
            print "attend", range(first_attend_index, first_compare_index)
            print "compare", range(first_compare_index, first_agg_index)
            print "aggregate", range(first_agg_index,len(trained_matrices))

            self.dimension_reducer = self._create_dimension_reducer(trained_matrices[1:first_attend_index])
            self.attend = self._create_attend(trained_matrices[first_attend_index:first_compare_index])
            self.compare = self._create_compare(trained_matrices[first_compare_index:first_agg_index])
            self.aggregate = self._create_aggregate(trained_matrices[first_agg_index:])
            self.params = {"E": trained_matrices[0]}
        else:
            self.pc = dy.ParameterCollection()
            self.dimension_reducer = self._create_dimension_reducer()
            self.attend = self._create_attend()
            self.compare = self._create_compare()
            self.aggregate = self._create_aggregate()
    
            self.params = {"E":  word_embeddings.as_dynet_lookup(self.pc)}
        
    def _create_dimension_reducer(self, params=None):
        return mat(self.pc, self.embedding_dim, self.hidden_dim)

    def _create_attend(self, params=None):
        activation = dy.tanh
        if params:
            attend = net.mlp_subnetwork.load(params, self.pc, activation, activation)
        else:
            attend = net.mlp_subnetwork(self.pc, [self.hidden_dim, self.hidden_dim, self.hidden_dim], activation, activation)
        return attend

    def _create_compare(self, params=None):
        activation = dy.rectify
        if params:
            compare = net.mlp_subnetwork.load(params, self.pc, activation, activation)
        else:
            compare = net.mlp_subnetwork(self.pc, [self.hidden_dim*2, self.hidden_dim, self.hidden_dim], activation, activation)
        return compare

    def _create_aggregate(self, params=None):
        hidden_activation = dy.rectify
        if params:
            agg = net.mlp_subnetwork.load(params, self.pc, hidden_activation, None)
        else:
            agg = net.mlp_subnetwork(self.pc, [self.hidden_dim*2, self.hidden_dim*2, self.classes_dim], hidden_activation, None)
        return agg

    def eval_loss(self, x, y, dropout=False):
        sentence_a, sentence_b = x
        
        sent_a_ords = [self.embeddings.get_ord(word) for word in sentence_a.split()]
        a_vecs = dy.lookup_batch(
            self.params["E"],
            sent_a_ords, 
            update=False
            )*0.05
       
        b_vecs = dy.lookup_batch(
            self.params["E"],
            [self.embeddings.get_ord(word) for word in sentence_b.split()], 
            update=False
            )*0.05

        output = self._eval_network(a_vecs, b_vecs, y, dropout=False)
        self.last_case_class = np.argmax(output.npvalue())
        loss = dy.pickneglogsoftmax(output, y)
        return loss


    def _eval_network(self, a_vecs, b_vecs, y, dropout):
        a_vecs_red = self.calc_reduce_dim(a_vecs)
        b_vecs_red = self.calc_reduce_dim(b_vecs)
        alphas, betas = self.calc_attend(a_vecs_red, b_vecs_red, dropout)
        v1_i, v2_j = self.calc_compare(a_vecs_red, b_vecs_red, alphas, betas, dropout)
        output = self.calc_aggregate(v1_i, v2_j,dropout)
        return output

    def calc_reduce_dim(self,vecs):
        return self.dimension_reducer.evaluate_network(vecs)

    def calc_attend(self, a_vecs, b_vecs, dropout):
        l_a = a_vecs.dim()[1]
        l_b = b_vecs.dim()[1]

        fa = self.attend.evaluate_network(a_vecs, True, dropout)
        fb = self.attend.evaluate_network(b_vecs, True, dropout) 

        e_ij = list()
        for i in range(l_a):
            e_ij.append(list())
            for j in range(l_b):
                e_ij[i].append(dy.dot_product(dy.pick_batch_elem(fa,i),dy.pick_batch_elem(fb,j)))
            
        beta_softmaxes = [dy.softmax(dy.concatenate(e_ij[i])) for i in range(l_a)]
        alpha_softmaxes = [dy.softmax(dy.concatenate([e_ij[i][j] for j in range(l_b)])) for i in range(l_a)]

        betas = [dy.esum(
            [dy.pick_batch_elem(b_vecs,j)*beta_softmaxes[i][j] for j in range(l_b)]
            ) for i in range(l_a)
            ]
        alphas = [dy.esum(
            [dy.pick_batch_elem(a_vecs,i)*alpha_softmaxes[i][j] for i in range(l_a)]
            ) for j in range(l_b)
            ]
        return alphas, betas
    
    def calc_compare(self, a_vecs, b_vecs, alphas, betas, dropout):
        ### not batched at the moment
        l_a = a_vecs.dim()[1]
        l_b = b_vecs.dim()[1]
        v1_i = [self.compare.evaluate_network(dy.concatenate([dy.pick_batch_elem(a_vecs,i), betas[i]]),True, dropout) for i in range(l_a)]
        v2_j = [self.compare.evaluate_network(dy.concatenate([dy.pick_batch_elem(b_vecs,j), alphas[j]]),True, dropout) for j in range(l_b)]
        
        return v1_i, v2_j

    def calc_aggregate(self, v1_i, v2_j, dropout):
        v1 = dy.esum(v1_i)
        v2 = dy.esum(v2_j)
        ret = self.aggregate.evaluate_network(dy.concatenate([v1,v2]),False, dropout)
        return ret

    def params_iterable(self):
        self_nets = [self.dimension_reducer, self.attend, self.compare, self.aggregate]
        yield self.params["E"]
        for net in self_nets:
            for param in net.params_iterable():
                yield param
    
    @classmethod
    def load(cls, word_embeddings, base_file, embedding_dim=None, hidden_dim=None, classes_dim=None):
        pc = dy.ParameterCollection()
        matrices = dy.load(base_file, pc)
        # matrices = matrices[1:] # for now, skip "E"
        return cls(word_embeddings, embedding_dim, hidden_dim, classes_dim, pc, matrices)

class mat:
    def __init__(self, pc, input_dim, output_dim):
        self.pc = pc
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = self.pc.add_parameters((self.output_dim, self.input_dim), init = 'uniform', scale=1) 
    
    def evaluate_network(self, x):
        return self.w * x
    
    def params_iterable(self):
        yield self.w
