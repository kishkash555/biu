import numpy as np
from collections import OrderedDict, Counter
from training import training_data #['pos_items','pos_counter','pos_counter1','pos_counter2','word_counts']
def initialize_test_data(c2str):
    pos_items = OrderedDict([('start',0),('oeuvre',1),('main',2),('side',3),('drink',4),('dessert',5)])
    pos_counter = np.array([30,15, 43, 30, 31,22])
    pos_counter1 = np.array([[0,10,10,4,4,2],[0,1,8,2,4,0],[0,1,9,10,8,12],[0,0,2,10,5,5],[0,3,11,4,4,1],[0,0,3,0,6,2]])
    word_counts = {
        'egg': np.array([0,1,0,0,3,0]),
        'salad': np.array([0,2,0,4,0,0]),
        'cookie': np.array([0,0,0,0,0,2]),
        'milk': np.array([0,0,0,0,5,0]),
        'cheese': np.array([0,4,0,4,0,1]),
        'soup': np.array([0,3,10,4,4,0]),
        'coke': np.array([0,0,0,3,8,1]),
        'ham': np.array([0,5,10,1,0,0]),
        'rice': np.array([0,0,3,4,0,0]),
        'quinoa': np.array([0,0,9,2,0,0]),
        'tuna': np.array([0,0,6,0,0,0]),
        'honey': np.array([0,0,0,0,3,5]),
        'nuts': np.array([0,0,2,0,0,2])
        }

    pos_counter2 = parse_counter2_str(c2str,list(pos_items.keys()))
    
    return training_data(pos_items, pos_counter,pos_counter1,pos_counter2,word_counts)

def parse_counter2_str(c2str,keys):
    ret = Counter()
    for row in c2str.split('\n'):
        digits, cnt = row.split()
        items = tuple([keys[int(d)] for d in digits])
        ret[items]=int(cnt)
    return ret






pos_counter2_str = """\
000	2
001	3
003	2
004	1
005	1
011	4
012	3
013	5
021	1
022	2
023	2
024	1
032	4
034	3
041	5
053	2
111	16
113	7
115	3
121	3
123	2
124	1
125	1
132	6
133	2
134	2
153	1
154	2
211	1
212	1
213	1
221	1
232	1
235	1
240	1
241	1
242	1
243	1
244	2
253	1
321	3
322	2
323	1
324	4
325	1
332	2
333	2
341	2
342	2
343	1
345	1
351	2
403	1
411	4
412	1
413	1
415	1
422	1
423	1
453	1
511	3
532	2
534	1
535	1"""

food_train_data = initialize_test_data(pos_counter2_str)