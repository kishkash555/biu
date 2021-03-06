import re
from collections import Counter
import config
#import memm_utils
import sys

def generate_triplets_from_tagged_sentence(tagged_sentence):
    padded_sentence = [('', config.start)] *2 + tagged_sentence + [('', config.stop)] *2
    for i in range(2,len(padded_sentence)-2):
        yield padded_sentence[i-2:i+3]

def triplet_to_feature_name_value_pairs(tagged_triplet, registered_features_l):
    for fe in registered_features_l:
        yield feature_str(*fe(tagged_triplet)) 

def generate_lines_with_tags(tagged_sentence, registered_features_l):
    for triplet in generate_triplets_from_tagged_sentence(tagged_sentence):
        features = ' '.join(sum(triplet_to_feature_name_value_pairs(triplet, registered_features_l),[]))
        if len(features):
            tag = triplet[2][1]
            yield tag + ' ' + features +'\n'

curr_word = lambda triplet: triplet[2][0]

def feature_str(feature_name, value):
    return [feature_name + "=" + str(value)] if value !=0 else []

def match_regex(compiled_exp, feature_name):
    def is_match(triplet, *argc):
        return feature_name, 1 if compiled_exp.fullmatch(curr_word(triplet).lower()) else 0
    return is_match

# def prev_tag(prev_tag):
#     def is_prev_tag(triplet, *argc):
#         return "prev_tag_"+prev_tag, 1 if triplet[1][1]==prev_tag else 0
#     return is_prev_tag
def prev_tag(triplet, *argc):
    return 'prev_tag_' + triplet[1][1], 1

# def previous_2_tags(tag1,tag2):
#     def are_last_2(triplet, *argc):
#         return (f'last_2_tags_{tag1}_{tag2}', 1 if triplet[0][1]==tag1 and triplet[1][1]==tag2 else 0)
#     return are_last_2
def previous_2_tags(triplet, *argc):
    return 'last_2_tags_' + triplet[0][1] + '_' + triplet[1][1], 1


def isnumber(word,*argc):
    try:
        float(curr_word(word))
        ret = 1
    except ValueError:
        ret = 0
    return "is_number",ret

def iscapitalized(word, *argc):
    return 'is_capitalized', (1 if word[2][0][0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' and word[1][0] != "*" else 0)

def vocab_predicates(vocab, ind=2):
    def is_word(triplet, *argc):
        word = triplet[ind][0].lower() 
        feature_name = "is_word_"+ (str(ind-2) if ind != 2 else '') + '_' + word
        word_is_number = isnumber(triplet)[1]==1
        if not word_is_number and (word in vocab or len(vocab)==0): 
            return feature_name, 1
        return feature_name, 0

    def get_prefix(triplet, *argc):
        cword = curr_word(triplet)
        word_is_number = isnumber(triplet)[1]==1 
        if word_is_number or cword in vocab or len(cword)<5:
            return 'prefix_', 0
        return 'prefix_'+cword[:4].lower(), 1
    
    def get_suffix(triplet, *argc):
        cword = curr_word(triplet) 
        word_is_number = isnumber(triplet)[1]==1 
        if word_is_number or cword in vocab or len(cword)<5:
            return 'suffix_', 0
        return 'suffix_'+cword[-4:].lower(), 1

    return is_word, get_prefix, get_suffix



known_words = set("the be to of and a in that have has had n't I it for not on with he as you do at this but his by from they we say her she or an will my one all\
    would there their what so up out if about who get which go me when make can like time no just him know take people into year your good some\
    could them see other than then now look only come its over think also back after use two how our work first well way even new want because any these give day most us\
    's $ % , a.m. p.m. ( ) ".split()) 

known_regexs = [
    (".+tion", 'tion'),
    ("[bcdfghjklmnpqrstvwxyz]{1,2}[aeiou].e",'lime_love_shave'),
    ("[bcdfghjklmnpqrstvwxyz]{1,2}[aeiou].ed",'limed_loved_shoved'),
    ("[a-z]{2,}ed",'ed'),
    (".{5,}nal",'directional'),
    ("(.{1,3}\.)+",'has_dots'),
    (".{4,}al",'al'),
    (".{4,}ality",'ality'),
    ("am|are|is|was|were|will|be|been",'be_conj'),
    ("have|has|had",'have_conj'),
    ('.{4,}ing','ing'),
    ('.{3,}ship','ship'),
    ('.{3,}ful','ful'),
    ('.{3,}fully','fully'),
    ('.+\-.*','hyphen'),
    ]

regexs = [match_regex(re.compile(ex), fn) for ex, fn in known_regexs]
registered_features = regexs + [isnumber, iscapitalized]  

def process_input_for_frequent_words(corpus_file, frequencies):
    word_counts = Counter()
    tag_counts = Counter()
    tag_pair_counts = Counter()
    word_re = re.compile('[a-z]{3,}')
    with open(corpus_file,'rt',encoding='utf8') as i:
        for line_in in i:
            words, tags = zip(*[w.rsplit("/",1) for w in line_in.split()])
            word_counts.update([w.lower() for w in words if word_re.fullmatch(w)])
            tag_counts.update(tags)
            tag_pair_counts.update(zip(tags[:-1],tags[1:]))
    ex_words = [w[0] for w in word_counts.most_common(frequencies.max_word_list_length) if w[1] >= frequencies.min_word_frequency and w[0] not in known_words]

    return set(ex_words)


def extract_features(corpus_file, feature_file):
    frequent_words = process_input_for_frequent_words(corpus_file, config.frequncies)
    vocab_words = frequent_words.union(known_words)
    is_word0, get_prefix, get_suffix = vocab_predicates(vocab_words)
    is_word_p, _, _ = vocab_predicates(vocab_words, 1)
    is_word_pp, _, _ = vocab_predicates(vocab_words, 0)

    is_word_n, _, _ = vocab_predicates(vocab_words, 3)
    is_word_nn, _, _ = vocab_predicates(vocab_words, 4)
    
    registered_features_l = registered_features + \
         [get_prefix, get_suffix, prev_tag, previous_2_tags] +\
         [is_word0, is_word_p, is_word_pp, is_word_n, is_word_nn]

    with open(corpus_file,'rt',encoding='utf8') as i:
        with open(feature_file,'wt', encoding='utf8') as o:
            c = 0
            for line_in in i:
                feature_generator = generate_lines_with_tags([tuple(w.rsplit("/",1)) for w in line_in.split()], registered_features_l)
                o.writelines(feature_generator)
                c += 1
                if c % 1000 ==0:
                    print(f"processed {c} lines of input.")
    return 0





if __name__ == '__main__':
    argv = sys.argv
    if len(argv)==1:
        print("Extract features running with default files")
        input_file = config.defaultFiles.tagged_input
        output_file = config.defaultFiles.memm_feature_out
    elif len(sys.argv !=3 ):
        print(f"usage: {sys.argv[0]} path_to_tagged_input_file path_to_feature_output_file")
        print("exiting.")
        exit()
    else:
        input_file = argv[1]
        output_file = argv[2]
    print(f"tagged input: {input_file}\n output:{output_file}")

    extract_features(input_file, output_file)
