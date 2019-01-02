from collections import Counter
from os import path

TRAIN_FNAME = '../snli_all.txt'
GLOVE_INPUT = '../glove.840B.300d.txt'
GLOVE_OUTPUT = '../glove_filtered.txt'

def load_train_words(lines):
    word_count = Counter()
    for line in lines:
        sentences = snli_sentence_extractor(line)
        for sentence in sentences:
            word_count.update([word for word in sentence.split()])
    print word_count.most_common(50)
    return word_count

def filter_glove(glove_lines, out_file, train_words):
    print "*********Starting over"
    rows = hits = 0
    for line in glove_lines:
        rows += 1
        if extract_glove_word(line) in train_words:
            out_file.write(line)
            hits +=1
        if rows % 50000 == 0:
            print "rows: {} hits: {}".format(rows, hits)
            out_file.flush()

def extract_glove_word(line):
    word = line.split()[0]
    if word == '<unk>': print 'unk found!'
    return word


def snli_sentence_extractor(line):
    return line.split('\t')[6:8]

def main(train_fname, glove_input_fname, glove_output_fname):
    root, ext = path.splitext(glove_output_fname)
    with open(train_fname) as t:
        t.readline() # skip header row
        word_count = load_train_words(t)

    num = 0
    augmented_output_fname = "{}{:02d}{}".format(root, num, ext)    
    while path.exists(augmented_output_fname):
        num += 1
        augmented_output_fname = "{}{:02d}{}".format(root, num, ext)

    print "writing to {}".format(augmented_output_fname) 
    word_count["<unk>"]=1   
    with open(augmented_output_fname,'wt') as out:
        with open(glove_input_fname, 'rt') as inp:
            filter_glove(inp, out, word_count)


if __name__ == "__main__":
    main(TRAIN_FNAME, GLOVE_INPUT, GLOVE_OUTPUT)
