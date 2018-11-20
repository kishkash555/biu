import sys
import hmm_main


if __name__ == "__main__":
    argv = sys.argv
    hmm_main.main(argv, 'viterbi')
    print("finished greedy tagging")

 