import _dynet as dy
import network as net
import numpy as np

class decomposable(net.network):
    def __init__(self, word_embeddings, embedding_dim=300, hidden_dim =200, classes_dim=3):
        self.embeddings = word_embeddings
        
        self.pc = dy.ParameterCollection()
        self.embedding_dim, self.hidden_dim, self.classes_dim = embedding_dim, hidden_dim, classes_dim

        self.params = {"E":  word_embeddings.as_dynet_lookup(self.pc)}
        self.dimension_reducer = self._create_dimension_reducer()
        self.attend = self._create_attend()
        self.compare = self._create_compare()
        self.aggregate = self._create_aggregate()
        
   
    def _create_dimension_reducer(self):
        red = net.mlp_subnetwork(self.pc, [self.embedding_dim, self.hidden_dim], lambda x: x, lambda x: x)
        return red
        
    def _create_attend(self):
        attend = net.mlp_subnetwork(self.pc, [self.hidden_dim, self.hidden_dim, self.hidden_dim], dy.rectify, dy.rectify)
        return attend

    def _create_compare(self):
        compare = net.mlp_subnetwork(self.pc, [self.hidden_dim*2, self.hidden_dim, self.hidden_dim], dy.rectify, dy.rectify)
        return compare

    def _create_aggregate(self):
        agg = net.mlp_subnetwork(self.pc, [self.hidden_dim*2, self.hidden_dim*2, self.classes_dim], dy.rectify, dy.pickneglogsoftmax)
        return agg

    def eval_loss(self, x, y, dropout=False):
        sentence_a, sentence_b = x
        
        sent_a_ords = [self.embeddings.get_ord(word) for word in sentence_a.split()]
        a_vecs = dy.lookup_batch(
            self.params["E"],
            sent_a_ords, 
            update=False
            )
       
        b_vecs = dy.lookup_batch(
            self.params["E"],
            [self.embeddings.get_ord(word) for word in sentence_b.split()], 
            update=False
            )

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
        return self.dimension_reducer.evaluate_network(vecs, False)

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
