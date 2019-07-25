#!/bin/bash
rm -rf ex4_submit
rm ex4_shahar.zip
mkdir ex4_submit
cp *.py ex4_submit/
cp test_y ex4_submit/
cp ex4_report.pdf ex4_submit/
echo "shahar siegman" > ex4_submit/details.txt
echo "011862141" >> ex4_submit/details.txt
zip ex4_shahar.zip ex4_submit/*

