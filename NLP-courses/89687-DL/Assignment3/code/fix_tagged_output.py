input_file = '../review/test4.ner'
output_file = '../review/test4_fixed.ner'
inp = open(input_file, 'rt')
out = open(output_file, 'wt')

states = ['sentence','blank'] # ,'skip line1','skip line1', 'first dot']
#dotdot=False
#line_skipped = False
state = 0
for line in inp:
    if state == 0: 
        if line[0] == '\n' :
            state = 1
    elif state == 1:
        if line[0] == '\n':
            pass
        else:
            line = ''
            state = 0
    out.write(line)

inp.close()
out.close()