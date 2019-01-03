#!/bin/bash
cd code
python strip_snli_columns.py ../snli_1.0/snli_1.0_train.txt ../snli_1.0/snli_1.0_train_stripped.txt
python strip_snli_columns.py ../snli_1.0/snli_1.0_dev.txt ../snli_1.0/snli_1.0_dev_stripped.txt
python strip_snli_columns.py ../snli_1.0/snli_1.0_test.txt ../snli_1.0/snli_1.0_test_stripped.txt
cd ..
