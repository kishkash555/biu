import test
food_train_data = test.food_train_data
import viterbi
import training

viterbi.trigram_viterbi(['egg','ham','cheese'],food_train_data)