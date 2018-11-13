import numpy as np
import config
from collections import Counter, namedtuple, defaultdict, OrderedDict


training_data = namedtuple('training_data',['pos_items','pos_counter','pos_counter1','pos_counter2','word_counts'])

def read_input(input_file_name): 
    """
    read tagged text of the form:
    The/DT luxury/NN auto/NN maker/NN last/JJ year/NN sold/VBD 1,214/CD cars/NNS in/IN the/DT U.S./NNP
    Howard/NNP Mosher/NNP ,/, president/NN and/CC chief/JJ executive/NN officer/NN ,/, said/VBD he/PRP anticipates/VBZ growth/NN for/IN the/DT luxury/NN auto/NN maker/NN in/IN Britain/NNP and/CC Europe/NNP ,/, and/CC in/IN Far/JJ Eastern/JJ markets/NNS ./.
    BELL/NNP INDUSTRIES/NNP Inc./NNP increased/VBD its/PRP$ quarterly/NN to/IN 10/CD cents/NNS from/IN seven/CD cents/NNS a/DT share/NN ./.
    The/DT new/JJ rate/NN will/MD be/VB payable/JJ Feb./NNP 15/CD ./.
    A/DT record/NN date/NN has/VBZ n't/RB been/VBN set/VBN ./.
    Bell/NNP ,/, based/VBN in/IN Los/NNP Angeles/NNP ,/, makes/VBZ and/CC distributes/VBZ electronic/JJ ,/, computer/NN and/CC building/NN products/NNS 

    and output counters encoded as numpy vectors:
    - pos_items: from part of speech to an ordinal (most common first)
    - pos_counter: a vector of counts based on the ordinals in pos_items
    - pos_counter1: a matrix of counts where the rows are the first in the pair and the columns are the second
    - pos_counter2: a sparse representation of (pos1, pos2, pos3) -> count
    - word_counts: a sparse representation of word -> vector of counts for each pos
    from part of speech(1) to part of speech(2) to count: the number of times sequence
    """

    with open(input_file_name,'rt',encoding='utf8') as a:
        file_as_string = a.read()

    # single counters
    word_pos_pairs = [tuple(w.rsplit("/",1)) for w in file_as_string.split()] #rsplit("/",1): split on the first slash from the right
    print("string length: {} words:{} ".format(len(file_as_string),len(word_pos_pairs)))
    pos_items_counts = Counter([pos for _,pos in word_pos_pairs]).most_common()
    pos_items = OrderedDict([(pos_count[0], i) for i, pos_count in enumerate(pos_items_counts)])
    pos_counter = np.array([c for _,c in pos_items_counts])
    
    # pair counter
    item_pairs = Counter([(pos_items[a[1]],pos_items[b[1]]) for a,b in zip(word_pos_pairs[:-1],word_pos_pairs[1:])])
    pos_counter1 = np.zeros((len(pos_items),len(pos_items)),np.int)
    for k,v in item_pairs.items():
        pos_counter1[k[0],k[1]]=v

    pos_counter2 = Counter([(a[1],b[1], c[1]) for a,b,c in zip(word_pos_pairs[:-2],word_pos_pairs[1:-1], word_pos_pairs[2:])])
    word_pos_counter = Counter(word_pos_pairs).most_common()
    #words = [word[0] for word in Counter([w for w,_ in word_pos_pairs]).most_common()]
    word_counts = defaultdict(lambda: np.zeros(len(pos_items),np.int))
    for w_p,c in word_pos_counter:
        w, p = w_p
        word_counts[w][pos_items[p]]=c

    return training_data(pos_items, pos_counter, pos_counter1, pos_counter2, word_counts)


def getQ(triplet, train_data):
    pos_items = train_data.pos_items
    pos_counter = train_data.pos_counter
    pos_counter1 = train_data.pos_counter1
    pos_counter2 = train_data.pos_counter2
    transition_lambdas = config.transition_lambdas
    t1,t2,t3 = triplet

    count_abc = pos_counter2[(t1,t2,t3)]  
    count_ab = pos_counter1[pos_items[t1],pos_items[t2]]
    count_bc = pos_counter1[pos_items[t2],pos_items[t3]]
    count_b = pos_counter[pos_items[t2]]
    count_c = pos_counter[pos_items[t3]]
    
    numwords = pos_counter.sum()        

    components = np.array([count_abc/count_ab, count_bc/count_b, count_c/numwords])
    ret = np.dot(transition_lambdas, components) 

    print(f'count_abc: {count_abc}\t count_ab: {count_ab}\ncount_bc: {count_bc}\tcount_b:{count_b}\ncount_c:{count_c}\tnumwods:{numwords}\nret:{ret}')
    return ret

def getLogQ(triplet, train_data):
    return -np.log(getQ(triplet, train_data))
def getEs(word, train_data):
    return train_data.word_counts[word]/train_data.pos_counter

def getLogEs(word, train_data):
    return -np.log(train_data.word_counts[word]) - np.log(train_data.pos_counter)
