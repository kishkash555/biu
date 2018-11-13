import numpy as np
import training
import config

def viterbi(N, q, e, T):
    """
    implementation of the viterbi algorithm.
    N - number of observations to generate
    q - transition matrix
    e - emission matrix
    T - number of states 
    """
    my_inf = 1000
    v = -my_inf * np.ones((T, N), np.double)
    bp = -np.ones((T,N), np.double)
    v[:,0] = 1.
    for j in range(1,N):
        for i in range(T):
            local_vec = v[:, (j-1)]*q[:,i]*e
            v[i,j] = np.max(local_vec)
            bp[i,j] = np.argmax(local_vec)
        print("J: {}\nV=\n{}\nbp=\n{}".format(j,v,bp))


def viterbi_sparse(edges,T):
    """
    edges: an array of lengh n, with triplets: (from_state, to_state, cost)
    see: http://www.cim.mcgill.ca/~latorres/Viterbi/va_alg.htm
    assumes the last layer has a single state.
    """
    S = []
    l_total = 0.
    next_L = np.zeros(T,np.double)
    for k, layer_edges in enumerate(edges):
        v = np.inf*np.ones((T,T),np.double)
        for c_k, c_k1, cost in layer_edges:
            v[c_k,c_k1] = next_L[c_k]+cost

        next_L=np.min(v,axis=0)
        S.append(np.argmin(v,axis=0))
        l_total += np.min(next_L)
        print("layer {}:v\n {}\nnext_L\n{}".format(k, v,next_L))
    path=[0]
    for step in S[::-1]:
        path = [step[path[0]]]+path
    return S, l_total, path


def trigram_viterbi(sentence, train_data):
    period = config.period
    pos_items = train_data.pos_items
    T = len(train_data.pos_items)
    V = np.zeros((T,T),np.double)
    V[pos_items[period],pos_items[period]] = 1
    bp = []
    for word in sentence:
        bp_n = -np.ones((T,T),np.int)
        e_w_r = training.getLogEs(word,train_data) # return a vector with entry per POS
        for t,b in enumerate(pos_items):
            q_r_tp = np.zeros((T,T),np.double)
            for r,c in enumerate(pos_items):
                search_triplets = [(tp,b,c) for tp in pos_items]
                q_r_tp[r,:] = np.array([training.getLogQ(triplet,train_data) for triplet in search_triplets])
            V_r_tp = q_r_tp + np.tile(e_w_r,(T,1)) + np.tile(V[t,:].T,(1,T))
            V_r = np.min(V_r_tp, axis=0)
            V[t,:] = V_r
            bp_n[t,:]=np.argmin(V_r_tp,axis=0)
        bp.append(bp_n)

    y_n = [np.argmin(V), np.argmin(V)]
    for bp_i in bp[2::-1]:
        y_n = bp_i[y[0],y[1]] + y_n
    return y_n

        