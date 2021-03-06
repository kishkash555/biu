#!/usr/bin/python2
import dynet as dy
import sys
import WordBindings as wb
import pickle
import numpy as np
import mlp
from os import path

PARAMS_FILE = 'params_267946'
INPUT_DIR = 'ner'
PREDICTIONS_FILE = '_predict'

def load_model(mode_file_name):
	m = dy.ParameterCollection()
	return [m] + list(dy.load(mode_file_name, m))


def predict_on_tuple(x_tuple, params, tags_array):
    output =  mlp.build_network(params, x_tuple)
    coded_tag = np.argmax(output.npvalue()) 
    return tags_array[coded_tag]


def test_stream_to_trainlike_stream(test_file,dummy_tag):
    for line in test_file:
        if len(line) <= 1:
            yield line
        else:
            yield line.strip() + " " + dummy_tag + "\n"

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
    trainlike = list(test_stream_to_trainlike_stream(test_file, tags_array[0]))
    sentence_tuples = wb.generate_train_5tuples(wb.train_stream_to_sentence_tuples(trainlike), word_dict, tag_dict, set())
    trainlike_row = iter(trainlike)
    for x_tuple, _ in sentence_tuples:
        prediction = predict_on_tuple(x_tuple, params, tags_array)
        current_row = next(trainlike_row)
        if len(current_row) <= 1:
            yield ""
            current_row = next(trainlike_row)
        word = current_row.split()[0]
        ret = word + " " + prediction
        yield(ret)

def load_dicts(dicts_file):   
    with open(dicts_file, 'rb') as f:     
        obj = pickle.load(f)
    word_dict = obj["word_dict"]
    tag_dict = obj["tag_dict"]
    return word_dict, tag_dict

    
if __name__ == "__main__":
    argv = sys.argv
    if len(argv) > 1 and argv[1] != "--":
        PARAMS_FILE = argv[1]
    if len(argv) > 2 and argv[2] != "--":
        wb.DICTS_FILE = argv[2]
    if len(argv) > 3 and argv[3] != "--":
        INPUT_DIR = argv[3]

    params = load_model(PARAMS_FILE)
    word_dict, tag_dict = load_dicts(wb.DICTS_FILE + '.' + INPUT_DIR )

    inp = open(path.join('..',INPUT_DIR,'test'),'rt')
    # out file will be created in current directory
    out = open(INPUT_DIR+PREDICTIONS_FILE,'wt') # this is simple string concatenation.
    for row in test_stream_to_tagged_stream(inp, word_dict, tag_dict):
        out.write(row+'\n')
    inp.close()
    out.close()

