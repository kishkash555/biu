import sys
import os
import utils

if __name__ == "__main__":
    input_file = sys.argv[1] # the untagged file
    model_file = sys.argv[2] # the model 
    feature_map_file = sys.argv[3] # dicphering the tags from numbers to human-readable tag names
    out_file = sys.argv[4]
    #todo: check all input files exist
    model = utils.load_model(input_file)



def 
