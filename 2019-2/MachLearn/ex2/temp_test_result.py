
good = [0,0,0]
total = 0
with open("train_y.txt",'rt') as y:
    with open("test_run.txt",'rt') as t:
        for line in t:
            r = []
            y_label = y.readline().split('.')[0]
            for i,l in enumerate(line.strip().split(',')):
                r.append(l[-1])
                #r.append('+' if l[-1]==y_label else '-')
                good[i] += l[-1]==y_label
            total += 1 
            print('{} {} {}'.format(line.strip(), ''.join(r), y_label))
print(','.join('{:.1%}'.format(g/total) for g in good))
