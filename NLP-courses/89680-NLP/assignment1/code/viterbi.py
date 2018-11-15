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


# def trigram_viterbi(sentence, train_data):
#     period = config.period
#     pos_items = train_data.pos_items
#     T = len(train_data.pos_items)
#     V = np.inf*np.zeros((T,T),np.double)
#     V[pos_items[period],pos_items[period]] = 0
#
#     bp = []
#     for word in sentence:
#         bp_n = -np.ones((T,T),np.int)
#         e_w_r = training.getLogEs(word,train_data) # return a vector with entry per POS
#         print(f'word is {word} logEs: {e_w_r}')
#         for t,b in enumerate(pos_items):
#             q_r_tp = np.zeros((T,T),np.double)
#             for r,c in enumerate(pos_items):
#                 search_triplets = [(tp,b,c) for tp in pos_items]
#                 q_r_tp[r,:] = np.array([training.getLogQ(triplet,train_data) for triplet in search_triplets])
#             e_tiled = np.tile(e_w_r,(T,1))
#             V_tiled = np.tile(V[:,t],(T,1)).T
#             V_r_tp = q_r_tp + e_tiled + V_tiled
#             V_r = np.min(V_r_tp, axis=1)
#             #print(f'q matrix: {q_r_tp}\nlatest search triplets:{search_triplets}\ne_tiled: {e_tiled}\nv_tiled:{V_tiled}\nresult:{V_r_tp}')
#
#             V[t,:] = V_r
#             bp_n[t,:]=np.argmin(V_r_tp,axis=1)
#             #print(f'\n\nV after iteration: {V}\nbp_n after iteration: {bp_n}')
#         print(f'\n\nV_t_r: {V}\nbp_n: {bp_n}')
#         bp.append(bp_n)
#
#     y_n = [np.argmin(V), np.argmin(V)]
#     for bp_i in bp[2::-1]:
#         y_n = [bp_i[y_n[0],y_n[1]]] + y_n
#     return y_n


def another_viterbi(sentence, train_data):
    # transition from start, start
    start = config.start
    states = list(train_data.pos_items.keys())
    T = len(states)
    bp_list = []

    first_word_transitions = np.array([training.getLogQ((start, start, s), train_data) for s in states])
    first_word_emissions = training.getLogEs(sentence[0],train_data)
    
    #print(f'first_word_transitions {first_word_transitions}')
    first_word_state_probabilities = first_word_transitions + first_word_emissions
    #first_word_state_probabilities = np.tile(first_word_state_probabilities,(T,1))
    print(f'first_word_state_probabilities {first_word_state_probabilities}')
    second_word_emissions = training.getLogEs(sentence[1],train_data)
    second_word_state_probabilities = -np.ones((T,T))
    for i in range(T):
        for j in range(T):
            second_word_state_probabilities[i,j]=\
                first_word_state_probabilities[i] + training.getLogQ((start,states[i],states[j]),train_data) + second_word_emissions[j]

    V_prev = second_word_state_probabilities
    #print(f'second word states:{second_word_state_probabilities} \
    #num finite: {(np.isfinite(second_word_state_probabilities)).sum()  }\
    #num finit vprev {(np.isfinite(second_word_state_probabilities)).sum()  }')
    V_list = [np.tile(first_word_state_probabilities,(T,1)), second_word_state_probabilities]
    for word in sentence[2:]:   
        word_emissions = training.getLogEs(word, train_data)
        V_current = -np.ones((T,T))
        bp_current = -np.ones((T,T), np.int)
        for i in range(T):
            for j in range(T):
                qs = np.array([training.getLogQ((state, states[i], states[j]),train_data) for state in states])
                if np.any(np.isinf(qs)):
                    print(f'infinity in qs {i},{j} word {word}')
                V_vec = V_prev[:,i] 
                Vqe= qs + V_vec + word_emissions[j]
                #print(f'V_vec: {V_vec}')
                if np.all(np.isinf(Vqe)):
                    V_current[i,j] = np.inf
                    bp_current[i,j] = -1
                else:
                    best_score_location = np.argmin(Vqe) 
                    V_current[i,j] = Vqe[best_score_location]
                    bp_current[i,j] = best_score_location
        if np.all(np.isinf(V_current)):
            print('infinity encountered')
        V_prev = V_current
        V_list.append(V_current)
        bp_list.append(bp_current)
    
    
    best_v_coord = np.unravel_index(np.argmin(V_current),(T,T))
    y = [best_v_coord[1], best_v_coord[0]]
    for i in range(len(bp_list)-1,-1,-1):
        y.append(bp_list[i][y[-1],y[-2]])
    y.reverse()
    return V_list, bp_list, y

    
        