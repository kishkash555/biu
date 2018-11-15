import config

import training
import viterbi
#trigram_viterbi = viterbi.trigram_viterbi
another_viterbi = viterbi.another_viterbi

train_data = training.read_input('..\\data\\ass1-tagger-train')

v, bp, y = another_viterbi(['Bell', 'makes','and','distributes','electronics'], train_data)

tags = list(train_data.pos_items.keys())
sentence_tags = [tags[yi] for yi in y]
print(sentence_tags)
