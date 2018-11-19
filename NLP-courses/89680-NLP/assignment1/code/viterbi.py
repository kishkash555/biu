import numpy as np
import scipy.sparse as sp
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

def viterbi_hmm(sentence, train_data):
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

        