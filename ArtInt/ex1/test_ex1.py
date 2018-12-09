#!/usr/bin/python
inputs = """\
    1 3 0-1-3-4-2-5-7-8-6
    3 4 1-2-3-4-5-6-0-8-7-9-11-12-13-10-14-15
    3 3 1-0-3-4-2-5-7-8-6
    3 3 8-6-7-2-5-4-3-0-1
    3 3 4-1-3-2-8-5-7-6-0
    2 3 0-1-3-4-2-5-7-8-6
    1 3 4-1-3-2-8-5-7-6-0
    3 3 0-2-6-4-7-1-5-8-3
    3 3 2-3-6-0-1-8-4-5-7
    3 3 5-2-8-4-1-7-0-3-6 # k
    3 3 0-1-2-3-4-5-6-7-8
    3 3 1-2-0-3-4-5-6-7-8"""

expected_outputs = """\
LULU 49 4
URRDLULDRRULULL 7259 15
ULU 4 3
RDLULDDRURULLDDRRUULLDRDRUULDLU 2360195 31
RDRDLULU 11 8
LULU 56 0
RDRDLULU 3342 8
ULLDRULURRDDLLURUL 7906 18
LULDDRRUULL 19 11
DLLURDLDRRUULDLURDDLUU 22075 22
LURULLDRRULDDLUURRDLLU 340494 22
UURDRULLDRRDLLURDRUULL 238609 22"""

from subprocess import call
for inp, expected_out in zip(inputs.split('\n'),expected_outputs.split('\n')):
    #if len(expected_out)>30:
    #    continue
    with open('input.txt','wt') as i:
        i.write('\n'.join(inp.strip().split()))
    call(['python3', 'ex1.py'])
    with open('output.txt','rt') as o:
        out = o.readline()
        print("\n\n{}\n{}\n{}\n".format(inp.strip(), out, expected_out))
