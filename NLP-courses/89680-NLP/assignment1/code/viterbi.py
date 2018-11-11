import numpy as np

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
    source: http://www.cim.mcgill.ca/~latorres/Viterbi/va_alg.htm
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

