import numpy as np

def viterbi(X, q, e, T):
    my_inf = 1000
    N = len(X)
    v = -my_inf * np.ones((T, N), np.double)
    bp = -np.ones((T,N), np.double)
    v[:,0] = 1.
    for j in range(1,len(X)):
        for i in range(T):
            v[i,j] = np.max(v[:, (j-1)]*q[:,i]*e)
            bp[i,j] = np.argmax(v[:, (j-1)]*q[:,i])
        print("J: {}\nV=\n{}\nbp=\n{}".format(j,v,bp))

