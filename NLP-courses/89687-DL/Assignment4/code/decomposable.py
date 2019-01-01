import _dynet as dy
import network as net


class decomposable:
    def __init__(self, word_embeddings, subnetworks_hidden_dims):
        self.embeddings = word_embeddings
        self.pc = dy.ParameterCollection()
        self.attend = self._create_attend()
        self.attend_hidden_dim, sel.compare_hidden_dim = subnetworks_hidden_dims

    def evaluate_network(self, sentences):
        embeddings = [[self.embeddings.get(word) for word in sentence.split()] for sentence in sentences]
        Eij = self.attend(embeddings)
    
    def _create_attend(self):
        attend = net.mlp_subnetwork(self.pc, [self.attend_hidden_dim]*2, dy.rectify, dy.rectify, True)
        return attend

    def _create_compare(self):
        compare = net.mlp_subnetwork(self.pc, [self.compare_hidden_dim]*2, dy.rectify, dy.rectify, True)
        return compare

    def calc_alphas_betas(self, a_vecs, b_vecs):
        l_a = len(a_vecs)
        l_b = len(b_vecs)

        fab = self.attend.evaluate_network(a_vecs+b_vecs, True) 
        e_ij = list()
        for i in range(l_a):
            e_ij.append(list())
            for b in range(l_b):
                e_ij[i].append(dy.dot(dy.lookup(fab,i, j+l_a)))
        
        beta_softmaxes = [dy.softmax(dy.concatenate(e_ij[i])) for i in range(l_a)]

        alpha_softmaxes = [dy.softmax(dy.concatenate([e_ij[i][j] for j in range(l_b)])) for i in range(l_a)]

        betas = [dynet.esum([dy.dot(beta_softmaxes[i],b_vecs[j]) for j in range(l_b)]) for i in range(l_a)]

        alphas = [dynet.esum([dy.dot(alpha_softmaxes[j],a_vecs[i]) for i in range(l_a)]) for j in range(l_b)]

        return alphas, betas
    
    def calc_compare(a_vecs, b_vecs, alphas,betas):

