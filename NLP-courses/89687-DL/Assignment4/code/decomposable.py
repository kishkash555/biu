import _dynet as dy
import network as net
import numpy as np

class decomposable(net.network):
    def __init__(self, word_embeddings, subnetworks_hidden_dims=(300,200,200)):
        self.embeddings = word_embeddings
        
        self.pc = dy.ParameterCollection()
        self.attend_hidden_dim, self.compare_hidden_dim, self.aggregate_hidden_dim = subnetworks_hidden_dims

        self.params = {"E":  word_embeddings.as_dynet_lookup(self.pc)}
        self.attend = self._create_attend()
        self.compare = self._create_compare()
        self.aggregate = self._create_aggregate()
        
   
    
    def _create_attend(self):
        attend = net.mlp_subnetwork(self.pc, [self.attend_hidden_dim]*2, dy.rectify, dy.rectify, True)
        return attend

    def _create_compare(self):
        compare = net.mlp_subnetwork(self.pc, [self.compare_hidden_dim]*2, dy.rectify, dy.rectify, False)
        return compare

    def _create_aggregate(self):
        agg = net.mlp_subnetwork(self.pc, [self.aggregate_hidden_dim]*2, dy.rectify, dy.pickneglogsoftmax, False)
        return agg

    def eval_loss(self, x, y):
        sentence_a, sentence_b = x
        print sentence_a, sentence_b
        
       
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

        print a_vecs, b_vecs
        output = self._eval_network(a_vecs, b_vecs, y)
        self.last_case_class = np.argmax(output.npvalue())
        loss = dy.pickneglogsoftmax(output, y)
        return loss


    def _eval_network(self, a_vecs, b_vecs, y):
        alphas, betas = self.calc_attend(a_vecs, b_vecs)
        v1_i, v2_j = self.calc_compare(a_vecs, b_vecs, alphas, betas)
        output = self.calc_aggregate(v1_i, v2_j)
        return output

    def calc_attend(self, a_vecs, b_vecs):
        l_a = a_vecs.dim()[1]
        l_b = b_vecs.dim()[1]
        print a_vecs.dim(), b_vecs.dim()
        print "evaluating attend network"
        #a_plus_b = dy.concatenate_to_batch([a_vecs[i] for i in range(l_a)] + [b_vecs[j] for j in range(l_b)])
        #print a_plus_b.dim()
        fa = self.attend.evaluate_network(a_vecs, True)
        fb = self.attend.evaluate_network(b_vecs, True) 
        print fa.dim(), fb.dim()
        
        print "checking batch elems", dy.pick_batch_elem(fa,0).dim()
        e_ij = list()
        for i in range(l_a):
            e_ij.append(list())
            for j in range(l_b):
                e_ij[i].append(dy.dot_product(dy.pick_batch_elem(fa,i),dy.pick_batch_elem(fb,j)))
            
        beta_softmaxes = [dy.softmax(dy.concatenate(e_ij[i])) for i in range(l_a)]
        print "beta npvalue:", beta_softmaxes[0].npvalue()
        print "beta pick", beta_softmaxes[0][0].npvalue(), beta_softmaxes[0][1].npvalue(), beta_softmaxes[0][2].npvalue()
        print "beta_softmaxes dim: {}".format(beta_softmaxes[0].dim())
        print "after pick_batch: {}".format(dy.pick_batch)
        alpha_softmaxes = [dy.softmax(dy.concatenate([e_ij[i][j] for j in range(l_b)])) for i in range(l_a)]
        print "alpha softmaxes dim: {}".format([x.dim() for x in alpha_softmaxes]) 

        betas = [dy.esum([dy.pick_batch_elem(b_vecs,j)*beta_softmaxes[i][j] for j in range(l_b)]) for i in range(l_a)]
        print "betas_dim", [x.dim() for x in betas]
        alphas = [dy.esum([dy.pick_batch_elem(a_vecs,i)*alpha_softmaxes[i][j] for i in range(l_a)]) for j in range(l_b)]

        print "alphas_dim", [x.dim() for x in alphas]
        
        return alphas, betas
    
    def calc_compare(self, a_vecs, b_vecs, alphas, betas):
        ### not batched at the moment
        l_a = len(a_vecs)
        l_b = len(b_vecs)
        v1_i = [self.compare.evaluate_network(dy.concatenate([a_vecs[i], betas[i]]),True) for i in range(l_a)]
        v2_j = [self.compare.evaluate_network(dy.concatenate([b_vecs[j], betas[j]]),True) for j in range(l_b)]
        
        return v1_i, v2_j

    def calc_aggregate(self, v1_i, v2_j):
        v1 = dy.esum(v1_i)
        v2 = dy.esum(v2_j)
        ret = self.aggregate.evaluate_network(dy.concatenate([v1,v2]),False)
        return ret
