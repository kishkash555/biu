

def read(fname):
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w,p = line
            sent.append((w,p))

pos_data = list(read('../pos/train'))
ner_data = list(read('../ner/train'))

pos_total = sum(len(x) for x in pos_data)
ner_total = sum(len(x) for x in ner_data)

print("pos: sentences {} words {}".format(len(pos_data),pos_total))
print("pos: sentences {} words {}".format(len(ner_data),ner_total))