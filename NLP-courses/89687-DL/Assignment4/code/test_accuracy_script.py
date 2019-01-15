if __name__ == "__main__":
    with open('../save/tagged5689.txt','rt') as lines:
        good = n_lines = 0
        for line in lines:
            last_two=line.strip()[-2:]
            if last_two[0]==last_two[1]:
                good += 1
            n_lines += 1
    print("good: {}, lines: {}, accuracy: {}".format(good, n_lines, float(good)/n_lines))

