#!/bin/bash
cat snli_1.0/snli_1.0_train.txt snli_1.0/snli_1.0_dev.txt snli_1.0/snli_1.0_test.txt > snli_all.txt
cd code
python filter_glove.py 
cd ..
rm snli_all.txt