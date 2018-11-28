#!/usr/bin/python2
import dynet as dy
import sys
import WordBindings as wb
import pickle
import numpy as np

FILE_NAME = 'params_713357'
def load_model(mode_file_name):
	m = dy.ParameterCollection()
	return dy.load(mode_file_name, m)


def predict_on_tuple(x_tuple, params, tags_array):
    output =  wb.build_network(params, x_tuple)
    coded_tag = np.argmax(output.npvalue()) 
    return tags_array[coded_tag]


def test_stream_to_trainlike_stream(test_file):
    for line in test_file:
        if len(line) <= 1:
            yield line
        else:
            yield line.strip()+" 0\n"

# def test_stream_to_coded_sentence(test_file, word_dict, tag_dict):
#     sentence = [('**START**', '')]*2
#     for line in test_file:
#         if len(line)>1:
#             word = line.strip()
#             if word in word_dict:
#                 sentence.append(word)
#             else:
#                 sentence.append('**UNK**')
#         elif len(sentence) > 2:
#             sentence += [('**STOP**', '')]*2
#             yield [word_dict(word) for word in sentence]
#             sentence = [('**START**', '')]*2

def coded_sentence_to_prediction_tuples(coded_sentence):
    for w in range(2,len(coded_sentence)-2):
        yield coded_sentence[w-2:w+3]

def test_stream_to_tagged_stream(test_file, word_dict, tag_dict):
    tags_array = list(tag_dict.keys())
    trainlike = list(test_stream_to_trainlike_stream(test_file))
    sentence_tuples = wb.generate_train_5tuples(wb.train_stream_to_sentence_tuples(trainlike), word_dict, tag_dict, 0)
    for x_tuple, _ in sentence_tuples:
        prediction = predict_on_tuple(x_tuple, params, tags_array)
        current_row = next(trainlike_row)
        if len(current_row) <= 1:
            yield "\n"
            current_row = next(trainlike_row)
        else:
            word = current_row.split()[0]
            ret = word + " " + prediction
            print(ret)
            yield(ret)
        

if __name__ == "__main__":
    argv = sys.argv
    if len(argv)>1 and argv[1] != "--":
        fname = argv[1]
    else:
        fname = FILE_NAME
    
    params = load_model(fname)
    