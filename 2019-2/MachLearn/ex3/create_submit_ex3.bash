#!/bin/bash
rm -rf ex3_submit
rm ex3_shahar.zip
mkdir ex3_submit
cp *.py ex3_submit/
# python predict_from_saved.py --model=save_model.pkl --data-x=test_x > test_y
cp test_y ex3_submit/
cp ex3_report.pdf ex3_submit/
echo "shahar siegman" > ex3_submit/details.txt
echo "011862141" >> ex3_submit/details.txt
zip ex3_shahar.zip ex3_submit/*

