# This file provides code which you may or may not find helpful.
# Use it if you want, or ignore it.
# import random
from collections import Counter
def read_data(fname):
    data = []
    with open(fname,"rt",encoding="utf8") as a: 
        for line in a:
            label, text = line.strip().lower().split("\t",1)
            data.append((label, text))
    return data

def text_to_bigrams(text):
    return ["%s%s" % (c1,c2) for c1,c2 in zip(text,text[1:])]

def text_to_unigrams(text):
    return [t for t in text]

def cleanup(text, cleanup_config):
    cc= cleanup_config
    if cc.short_strings and len(text) < cc.short_strings:
        return ""
    words = text.split(" ")
    words = [w for w in words if w!=" " and len(w)>0]
    if cc.at_sign:
        words = [w for w in words if w[0]!="@"]
    if cc.repeating_character:
        words = [w for w in words if Counter(w).most_common(1)[0][1] < cc.repeating_character]
    if cc.remove_urls:
        words = [w for w in words if len(w)<4 or w[:4]!='http']
    return " ".join(words)

def create_ngram_vocab(data, max_count, text_to_ngrams):
    ngrams = Counter()
    for _, text in data:
        ngrams.update(text_to_ngrams(text))
    return [i for i,j in ngrams.most_common(max_count) ]

def save_strings_to_file(fname, strings):
    with open(fname,'wt',encoding='utf8') as a:
        for s in strings:
            a.write(s+'\n')



if __name__ == "__main__":
    TRAIN = [(l,text_to_bigrams(t)) for l,t in read_data("train")]
    DEV   = [(l,text_to_bigrams(t)) for l,t in read_data("dev")]

    from collections import Counter
    fc = Counter()
    for l,feats in TRAIN:
        fc.update(feats)

    # 600 most common bigrams in the training set.
    vocab = set([x for x,c in fc.most_common(600)])

    # label strings to IDs
    L2I = {l:i for i,l in enumerate(list(sorted(set([l for l,t in TRAIN]))))}
    # feature strings (bigrams) to IDs
    F2I = {f:i for i,f in enumerate(list(sorted(vocab)))}

