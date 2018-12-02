import sys

if __name__ == "__main__":
    with open(sys.argv[1],'rt', encoding='utf8') as i:
        with open(sys.argv[2], 'wt', encoding='utf8') as o:
            for line in i:
                words = [word.rsplit('/',1)[0] for word in line.split()]
                o.write(' '.join(words)+'\n')
