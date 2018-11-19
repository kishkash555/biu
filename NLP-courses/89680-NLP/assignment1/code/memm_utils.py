import numpy as np

def check_accuracy(test_file, output_file):
    with open(test_file,'rt',encoding='utf8') as t:
        with open(output_file,'rt',encoding='utf8') as o:
            correct_counter, total_counter = 0 , 0
            while True:
                lt = t.readline()
                lo = o.readline()
                if lt == '':
                    if lo != '':
                        print("hit EOF of test first")
                    break
                test_words, test_tags = zip(*[w.rsplit('/',1) for w in lt[:-1].split()])
                out_words, out_tags = zip(*[w.rsplit('/',1) for w in lo[:-1].split()])
                if all([w1==w2 for w1, w2 in zip(test_words, out_words)]):
                    correct_counter += sum([1 for t1, t2 in zip(test_tags, out_tags) if t1==t2])
                    total_counter += len(out_words)
                else:
                    print(f"lines don't match:\n{' '.join(out_words)}\n{' '.join(test_words)}")
                    return
    print(f"correct: {correct_counter} of {total_counter}, {100*correct_counter/total_counter}% ")
    return correct_counter, total_counter


def load_map_file(feature_map_file):
    with open(feature_map_file,'rt',encoding='utf8') as mf:
        tag_dict = dict()
        feature_dict = dict()
        for line in mf:
            name, ind = line.split()
            name, value = name.split('=')
            if name == 'tag':
                tag_dict[int(ind)] = value 
            else:
                feature_dict[name] = int(ind)  
            
    return tag_dict, feature_dict

def get_registered_features(feature_names):
    registered_words = []
    registered_tags = []
    registered_tag_pairs = []
    registered_prefixes = []
    registered_suffixes = []
    is_word = "is_word__"
    iwl = len(is_word)
    prev_tag= "prev_tag_"
    ptl = len(prev_tag)
    prev_2 = "last_2_tags_"
    p2l = len(prev_2)
    prefix = "prefix_"
    pxl = len(prefix)
    suffix = "suffix_"
    sxl = len(suffix)
    for name in feature_names:
        if name.startswith(is_word):
            registered_words.append(name[iwl:])
        elif name.startswith(prev_tag):
            registered_tags.append(name[ptl:])
        elif name.startswith(prev_2):
            tag1,tag2 = name[p2l:].split("_")
            registered_tag_pairs.append((tag1,tag2))
        elif name.startswith(prefix):
            registered_prefixes.append(name[pxl:])
        elif name.startswith(suffix):
            registered_suffixes.append(name[sxl:])
    return registered_words, registered_tags, registered_tag_pairs, registered_prefixes, registered_suffixes

