from collections import Counter, OrderedDict
import config
import sys

def process_input_for_tags_and_features(feature_file):
    tag_counter=Counter()
    feature_name_counter = Counter()
    with open(feature_file,'rt',encoding='utf8') as i:
         for line_in in i:
             tokens = line_in.split()
             tag_counter.update(tokens[:1])
             feature_name_counter.update(s.split('=')[0] for s in tokens[1:])
    tag_dict = OrderedDict([(a[1][0],a[0]) for a in enumerate(tag_counter.most_common())])
    feature_name_dict = OrderedDict([(a[1][0],a[0]) for a in enumerate(feature_name_counter.most_common())])
    return tag_dict, feature_name_dict


def write_features(feature_file, feature_vecs_file, tag_dict, feature_name_dict):
    with open(feature_file,'rt',encoding='utf8') as i:
        with open(feature_vecs_file,'wt',encoding='utf8') as o:
            for line_in in i:
                tokens_in = line_in.split()
                tag = str(tag_dict[tokens_in[0]])
                feature_codes = sorted([feature_name_dict[f.split('=')[0]] for f in tokens_in[1:]])
                tokens_out = [tag] + ['{}:1'.format(fe) for fe in feature_codes]
                o.write(' '.join(tokens_out)+'\n')
    return 0

def write_map(map_file, tag_dict, feature_name_dict):
    with open(map_file,'wt',encoding='utf8') as o:
        o.writelines([f'tag={t} {d}\n' for t,d in tag_dict.items()])
        o.writelines([f'{f}=1 {d}\n' for f,d in feature_name_dict.items()])


def convertFeatures(feature_file, feature_vecs_file, map_file):
    tag_dict, feature_name_dict = process_input_for_tags_and_features(feature_file)
    write_features(feature_file, feature_vecs_file, tag_dict, feature_name_dict)
    write_map(map_file, tag_dict, feature_name_dict)
    return 0

if __name__ == "__main__":
    argv = sys.argv
    if len(argv)==1:
        print("Convert features running with default files")
        input_file = config.defaultFiles.memm_feature_out
        feature_vecs_file = config.defaultFiles.memm_feature_vec
        map_file = config.defaultFiles.memm_feature_map
    elif len(sys.argv !=4 ):
        print(f"usage: {sys.argv[0]} path_to_feature_file path_to_feature_vecs_file path_to_map_file")
        print("exiting.")
        exit()
    else:
        input_file = argv[1]
        feature_vecs_file = argv[2]
        map_file = argv[3]
    print(f"parameters:\n\tinput: {input_file}\n\tfeature vectors:{feature_vecs_file}\n\tmap file:{map_file}")

    convertFeatures(input_file, feature_vecs_file, map_file)
