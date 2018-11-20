import numpy as np
import scipy.sparse as sp
import hmm 
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



def viterbi_hmm(sentence_str, train_data):
    # transition from start, start
    start = config.start
    states = list(train_data.pos_items.keys())
    T = len(states)
    bp_list = []
    sentence = sentence_str.split()
    first_word_transitions = np.array([hmm.getLogQ((start, start, s), train_data) for s in states])
    first_word_emissions = hmm.getLogEs(sentence[0],train_data)
    
    #print(f'first_word_transitions {first_word_transitions}')
    first_word_state_probabilities = first_word_transitions + first_word_emissions
    #first_word_state_probabilities = np.tile(first_word_state_probabilities,(T,1))
    #print(f'first_word_state_probabilities {first_word_state_probabilities}')
    second_word_emissions = hmm.getLogEs(sentence[1],train_data)
    second_word_state_probabilities = -np.ones((T,T))
    for i in range(T):
        for j in range(T):
            second_word_state_probabilities[i,j]=\
                first_word_state_probabilities[i] + hmm.getLogQ((start,states[i],states[j]),train_data) + second_word_emissions[j]

    V_prev = second_word_state_probabilities
    #print(f'second word states:{second_word_state_probabilities} \
    #num finite: {(np.isfinite(second_word_state_probabilities)).sum()  }\
    #num finit vprev {(np.isfinite(second_word_state_probabilities)).sum()  }')
    # V_list = [np.tile(first_word_state_probabilities,(T,1)), second_word_state_probabilities]
    for word in sentence[2:]:   
        word_emissions = hmm.getLogEs(word, train_data)
        V_current = -np.ones((T,T))
        bp_current = -np.ones((T,T), np.int)
        for i in range(T):
            for j in range(T):
                qs = np.array([hmm.getLogQ((state, states[i], states[j]),train_data) for state in states])
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
       # V_list.append(V_current)
        bp_list.append(bp_current)
        print(word)
    
    best_v_coord = np.unravel_index(np.argmin(V_current),(T,T))
    y = [best_v_coord[1], best_v_coord[0]]
    for i in range(len(bp_list)-1,-1,-1):
        y.append(bp_list[i][y[-1],y[-2]])
    y.reverse()
    ret = [states[i] for i in y]
    print(' '.join(['/'.join(a) for a in zip(sentence, ret)]))
    return ret



# , model, registered_features, feature_dict
def viterbi_memm(sentence_str, states, get_triplet_scores):
    # transition from start, start
    start = config.start
    states = [start] + states
    T = len(states)
    start_ind = 0
    bp_list = []
    padded_sentence = ['']*2 + sentence_str.split() + ['']*2
    n_words = len(padded_sentence) - 4
    V_prev = np.inf * np.ones((T,T))
    V_prev[start_ind, start_ind] = 0
    V_list =[]
    for w in range(n_words):
        print(w)
        current_triplet = [[padded_sentence[k],''] for k in range(w,w+5)]
        probas = np.inf * np.ones((T,T))
        V_current = np.inf * np.ones((T,T))
        bp_current = -np.ones((T,T),np.int)
        for i in range(T):
            current_triplet[1][1] = states[i]
            for tp in range(T):
                current_triplet[0][1] = states[tp]
                probas[tp,1:] = get_triplet_scores(current_triplet)
                if w <= 1:
                    break
            for j in range(T):
                v_vec = probas[:,j] + V_prev[:,i]
                best_score_location = np.argmin(v_vec)
                V_current[i,j] = v_vec[best_score_location]
                bp_current[i,j] = best_score_location
            if w == 0:
                break
        V_prev = V_current
        V_list.append(V_current)
        bp_list.append(bp_current)

    best_v_coord = np.unravel_index(np.argmin(V_current),(T,T))
    y = [best_v_coord[1], best_v_coord[0]]
    for i in range(len(bp_list)-1,1,-1):
        y.append(bp_list[i][y[-1],y[-2]])
    y.reverse()
    ret = [states[i] for i in y]
    print("sentence done.")
    return ret

from collections import OrderedDict, defaultdict

def viterbi_prune(sentence_str, states, get_triplet_scores):
    start = config.start
    states = [start] + states
    T = len(states)
    start_ind = 0
    prune_keep_count = config.viterbi_prune_keep_count
    bp_list = []
    padded_sentence = ['']*2 + sentence_str.split() + ['']*2
    n_words = len(padded_sentence) - 4
    V_prev = OrderedDict()
    V_prev[(start_ind,start_ind)] = 0
    i_to_tp_prev = defaultdict(set)
    j_to_tp = defaultdict(set)
    i_to_tp_prev[start_ind].add(start_ind)
    for w in range(n_words):
        current_triplet = [[padded_sentence[k],''] for k in range(w,w+5)]
        probas = OrderedDict()
        V_current = OrderedDict()
        # i_to_tp_current = defaultdict(set)
        bp_current = OrderedDict()
        for tp, i in V_prev.keys():
            current_triplet[0][1] = states[tp]
            current_triplet[1][1] = states[i]
            scores = get_triplet_scores(current_triplet)
            for j in range(T-1):
                probas[(tp,j+1)] = scores[0,j]
                j_to_tp[j+1].add(tp)
        
        for i in i_to_tp_prev:
            for j in j_to_tp:
                active_coord = i_to_tp_prev[i].intersection(j_to_tp[j])
                curr_min = np.inf
                for coord in active_coord:
                    curr = probas[(coord,j)] + V_prev[(coord,i)]
                    if  curr < curr_min:
                        curr_min = curr
                        curr_argmin = coord
                bp_current[(i,j)] = curr_argmin
                V_current[(i,j)] = curr_min
                # i_to_tp_current[j].add(i)
        # now for some pruning: we keep only a few "most promising" entries in V and dump the rest
        v_spanned_sorted = sorted(V_current.items(), key = lambda x: x[1])

        V_prev = OrderedDict(v_spanned_sorted[:prune_keep_count])
        i_to_tp_prev = defaultdict(set)
        for i, j in V_prev.keys():
            i_to_tp_prev[j].add(i)

        bp_list.append(bp_current)

    curr_min = np.inf
    for coord, val in V_current.items():
        if val < curr_min:
            curr_min = val
            min_coord = coord
    y = [min_coord[1],min_coord[0]]
    for i in range(len(bp_list)-1, 1, -1):
        y.append(bp_list[i][(y[-1],y[-2])])

    y.reverse()
    ret = [states[i] for i in y]
    # print("sentence done.")
    return ret

        