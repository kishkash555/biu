import numpy as np
K = 5

def load_E():
    E = np.fromfile('wordVectors.txt', sep=' ')
    E = E.reshape(E.shape[0]/50,50)
    return E

def load_vocab():
    with open('vocab.txt','rt') as a:
        word_list = a.readlines()
    return [w.strip() for w in word_list]


def caluclate_most_similar_k(E, vocab, words_to_test, k):
    # all we do is compute cosines so let's go ahead and normalize E
    nn=np.sum(E**2,axis=1)**(1./2)
    normalized_E = E / nn[:,np.newaxis]  #np.newaxis adds a 1-dimension to the vector
    # this dictionary keeps track of the index of each word_to_test in E
    vocab_to_ind1 = dict((w, i) for i, w in enumerate(vocab) if w in set(words_to_test))
    # this dictionary keeps track of the index of each word_to_test in cosines
    vocab_to_ind2 = dict((w,i) for i, w in enumerate(vocab_to_ind1.keys()))
    # extract the (already normalized) vectors of the words_to_test
    words_to_test_vectors = normalized_E[list(vocab_to_ind1.values()),:]
    # cosines(i,j) is the cosine distance between word i in the original vocab and word j (number of columns = number of words to test)
    cosines = 1 - normalized_E.dot(words_to_test_vectors.T)
    # k_most_similar(i,j) is the i'th most similar word to word j (out of words to test). the top one is always itself.
    k_most_similar = np.argsort(cosines,axis=0)[:k+1,:]

    # construct a dictionary word_to_test -> list of tuples [(similar_word, cosine_similarity),...]
    ret = {}
    for word in words_to_test:
        similar_words_indices = k_most_similar[1:,vocab_to_ind2[word]] # the first entry is always the word itself
        similar_word_words = [vocab[i] for i in similar_words_indices]
        similar_words_distances = np.round(1 - cosines[similar_words_indices, vocab_to_ind2[word]],3)
        ret[word] = zip(similar_word_words, similar_words_distances)
    return ret

if __name__=="__main__":
    E = load_E()
    vocab = load_vocab()
    words_to_test = 'dog,england,john,explode,office'.split(',')
    k_similar = caluclate_most_similar_k(E,vocab,words_to_test, K)
    for word in words_to_test:
        print("similar to {}: {}".format(word,k_similar[word]))
