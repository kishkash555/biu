import sys
#out_file = 'ex2_24da3d_out.txt'
out_file = sys.argv[1]

if __name__ == "__main__":
    gold = open('gold.txt','rt')
    test = open(out_file,'rt')

    lines, tested, g1, g2 = 0 ,0, 0 ,0
    while True:
        lines += 1
        gold_line = gold.readline().split('\t')
        test_line = test.readline().split(' - ')
        if len(gold_line[0])==0:
            break
        if gold_line[0]!=test_line[0]:
            ValueError("file name mismtach")
        gd = int(gold_line[1])
        t1, t2 = int(test_line[1]), int(test_line[2])
        if gd > 0:
            tested += 1
            g1 = g1 + (1 if t1==gd else 0)
            g2 = g2 + (1 if t2==gd else 0)
        
    print('lines: {} tested: {} good1: {} ({:.1%}) good2: {} ({:.1%})'.format(lines, tested, g1, g1/tested, g2, g2/tested))