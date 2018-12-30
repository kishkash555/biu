import algos

train_file = 'train.txt'
test_file = 'test.txt'
def read_data():
    with open(train_file,'rt', encoding='utf8') as a:
        train_data = algos.read_input(a)   
    with open(test_file,'rt', encoding='utf8') as a:
        test_data = algos.read_input(a)   
    return train_data, test_data

def test_knn():
    train_data, test_data = read_data()
    nearest5 = algos.knn(5, algos.hamming_distance)
    nearest5.fit(train_data)
    labels = nearest5.predict(test_data)
    good = 0
    for t, l in zip(test_data, labels):
        if t[-1]==l:
            good += 1
    print('good: {}, overall: {}'.format(good, len(test_data)))

def test_decision_tree():
    train_data, _ = read_data()
    dt = algos.decision_tree()
    dt.fit(train_data)
    print(dt.tree)
    1

if __name__ == "__main__":
    test_decision_tree()