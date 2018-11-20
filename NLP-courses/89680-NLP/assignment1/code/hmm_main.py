import hmm
import viterbi
import config

def main(argv, decoder):
    decoder = decoder.lower()
    if decoder == 'greedy':
        decoder_func = hmm.greedy_hmm_tagger
    elif decoder == 'viterbi':
        decoder_func = viterbi.viterbi_hmm
    else:
        print(f"decoder should be 'greedy' or 'viterbi', not {decoder}")
        return
    if len(argv)==1:
        print("GreedyTag running with default files")
        input_file = config.defaultFiles.untagged_test
        qmle_file = config.defaultFiles.qmle
        emle_file = config.defaultFiles.emle
        out_file = config.defaultFiles.hmm_output
    elif len(argv) < 4:
        print(f"usage: {argv[0]} untagged_input_file qmle_file emle_file output_file")
        print("exiting.")
        exit()
    else:
        input_file = argv[1]
        qmle_file = argv[2]
        emle_file = argv[3]
        out_file = argv[4]
    print(f"the following will be used:\n\tinput: {input_file}\n\tqmle: {qmle_file}\n\temle: {emle_file}\n\toutput: {out_file}")

    train_data = hmm.load_train_data_from_mle_files(qmle_file, emle_file)
    with open(input_file,'rt',encoding='utf8') as i:
        with open(out_file,'wt',encoding='utf8') as o:
            for line in i:
                tagged_sentence = decoder_func(line, train_data)
                tagged_str = ' '.join('/'.join(p) for p in tagged_sentence)
                o.write(tagged_str+'\n')
