
def parse_report(lines):
    nums = []
    for line in lines:
        line = line.strip().split(' ')
        if len(line) < 10:
            continue
        if not (all([a==b for a, b in zip([line[i] for i in [2,3,4,6]], 'average loss after iterations:'.split(' '))])):
            continue
        nums.append([float(x) for x in [line[i] for i in [5,7,9,12]]])

    iterations, loss, train_acc,  dev_acc = zip(*nums)
    return iterations, loss, train_acc,  dev_acc

