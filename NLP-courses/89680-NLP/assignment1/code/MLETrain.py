import hmm
import sys
import config

if __name__ == "__main__":
    argv = sys.argv
    if len(argv)==1:
        print("MLETrain running with default files")
        input_file = config.defaultFiles.tagged_input
        qmle_file = config.defaultFiles.qmle
        emle_file = config.defaultFiles.emle
    elif len(sys.argv != 4):
        print(f"usage: {sys.argv[0]} path_to_tagged_input_file qmle_output_file emle_output_file")
        print("exiting.")
        exit()
    else:
        input_file = argv[1]
        qmle_file = argv[2]
        emle_file = argv[3]
    print(f"the following will be used:\n\tinput: {input_file}\n\tqmle: {qmle_file}\n\temle: {emle_file}")
    train_data = hmm.read_input(input_file)
    with open(qmle_file,'wt', encoding='utf8') as q:
        c = 0
        for line in hmm.q_mle_output(train_data):
            c += 1
            q.write(line+'\n')
    print(f'qmle: wrote {c} lines')
    with open(emle_file, 'wt', encoding='utf8') as e:
        c=0
        for line in hmm.e_mle_output(train_data):
            c+=1
            e.write(line+'\n')
    print(f'emle: wrote {c} lines')
    