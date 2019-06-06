
def prob_calc_bf(y, word, alphabet):
    """
    brute force probability calculation
    y: the matrix of probabilities per time step, note it has to be TRANSPOSED before the call to prob_calc_bf
    word: the word to calculate, eg. "cat"
    alphabet: the interpretation of each column in y, eg. 'abcdef'
    note the last column represents the probability of blank
    """
    time_steps = y.shape[1]
    alphabet_dict = { t: i for i, t in enumerate(alphabet)}
    alphabet_dict[':'] = len(alphabet_dict)
    ret = 0.
    for path in generate_paths('',0,word, time_steps):
        path_p = 1
        for i, char in enumerate(path):
            path_p *= y[alphabet_dict[char],i]
        ret += path_p
    return ret
    
def generate_paths(current_path, next_symbol, word, total_length):
    """
    recursive generation of all valid paths for a given word and a total length
    e.g. "cat", with total length of 5 will generate 'cat::','ca:t:','c:a:t','ca::t','c::at', '::cat',
    'cccat', 'ccaatt' etc.
    current path: always '', except within recursive call
    next_symbol: always 0, except within recursive call
    word: the word to generate paths for
    total_length: the total length of the strings to generate
    note this function is a _generator_. correct usages:
    for p in generate_paths(...)
    or
    paths = list(generate_paths(...))
    """
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
    """
    returns the word that is represented by a sequence with blanks and duplications.
    set(map(squeeze, generate_path(...))) will return a set with just the original word since all sequences
    represent the same original word
    """
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
