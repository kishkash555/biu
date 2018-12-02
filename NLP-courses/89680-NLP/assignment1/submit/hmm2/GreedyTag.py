import sys
import hmm_main


if __name__ == "__main__":
    argv = sys.argv
    hmm_main.main(argv, 'greedy')
    print("finished greedy tagging")

 