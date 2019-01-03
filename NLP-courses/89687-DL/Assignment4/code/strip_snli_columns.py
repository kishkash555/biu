import sys
import parsers

if __name__ == "__main__":
    argv = sys.argv
    data, _ = parsers.load_snli(argv[1])
    with open(argv[2], 'wt') as a:
        parsers.dump_snli(data, a)