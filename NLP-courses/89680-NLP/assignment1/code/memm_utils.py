import numpy as np

def sentence_log_probas(model, sentence):
    """
    return a numpy 2d array of neg-log probas (always positive numbers) for use as input to viterbi / greedy
    """
    words = sentence.split()
    ll = len(words)


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