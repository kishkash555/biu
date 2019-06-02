
def prob_calc_bf(y, word, alphabet):
    label_length = len(word)
    alpha = np.zeros(label_length * 2 + 1)
    alphabet_dict = { t: i for i, t in enumerate(alphabet)}
    blank_ordinal = len(alphabet)

def generate_paths(current_path, next_symbol, word, total_length):
    if next_symbol == len(word):
        yield current_path + ':'*(total_length-len(current_path))
        return 
    if len(current_path) == total_length:
        yield current_path
        return 
    if len(current_path) > total_length:
        return
    remaining_symbols = len(word) - next_symbol
    remaining_steps = total_length - len(current_path)
    if remaining_symbols < remaining_steps:
        if current_path=='' or current_path[-1] != word[next_symbol]:
            for p in generate_paths(current_path+':', next_symbol, word, total_length):
                yield p
        for p in generate_paths(current_path+word[next_symbol], next_symbol, word, total_length):
            yield p
    if next_symbol +1 < len(word) and word[next_symbol+1] == word[next_symbol]:
        if remaining_symbols < remaining_steps:
            for p in generate_paths(current_path+word[next_symbol]+':', next_symbol+1, word, total_length):
                yield p
    else:
        for p in generate_paths(current_path+word[next_symbol], next_symbol+1, word, total_length):
            yield p

def squeeze(pt):
    ret = []
    curr = '*'
    for c in pt:
        if c != ':' and c != curr:
            ret.append(c)
        curr=c
    return ''.join(ret)

if __name__ == "__main__":
    pts = list(generate_paths('',0,'aaba',5))
    print(set(map(squeeze,pts)))
    print(len(pts), len(set(pts)))
    
    for p in sorted(pts):
        print(p)
