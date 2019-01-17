import algos

train_file = 'train.txt'
test_file = 'test.txt'
def read_data():
    with open(train_file,'rt', encoding='utf8') as a:
        train_data = algos.read_input(a)   
    with open(test_file,'rt', encoding='utf8') as a:
        test_data = algos.read_input(a)   
    return train_data, test_data

def test_knn(train_data, test_data):
    nearest5 = algos.knn(5, algos.hamming_distance)
    nearest5.fit(train_data)
    predictions = nearest5.predict(test_data)
    # good = 0
    # for t, l in zip(test_data, labels):
    #     if t[-1]==l:
    #         good += 1
    # print('good: {}, overall: {}'.format(good, len(test_data)))
    return predictions

def test_decision_tree(train_data, test_data, the_tree = None):
    dt = algos.decision_tree()
    dt.fit(train_data)
    # print(dt.tree)
    predictions = dt.predict(test_data)
    if type(the_tree) == list:
        the_tree.append(dt)
    return predictions

def test_naive_bayes(train_data, test_data):
    nb = algos.naive_bayes()
    nb.fit(train_data)
    predictions = nb.predict(test_data)
    return predictions

def create_report(predictions, actual):
    yield "Num\tDT\tKNN\tnaiveBase"
    n_classes = len(predictions[0])
    n_cases = len(predictions)
    good = [0] * n_classes
    for i, (pred, act) in enumerate(zip(predictions, actual),1):
        yield "\t".join([str(i)]+list(pred))
        good = [g+ int(p==act) for g,p in zip(good, pred)]
    yield "\t"+"\t".join(['{:.2f}'.format(g/n_cases) for g in good])





if __name__ == "__main__":
    train_data, test_data = read_data()
    the_tree = []
    pred=[
        test_decision_tree(train_data, test_data, the_tree),
        test_knn(train_data, test_data),
        test_naive_bayes(train_data, test_data)
    ]

    with open('output.txt','wt') as outf:
        for row in create_report(list(zip(*pred)),[t[-1] for t in test_data]):
            outf.write(row+'\n')

    with open('output_tree.txt', 'wt') as outf:
        for row in the_tree[0].create_tree_structure_report():
            outf.write(row+'\n')



