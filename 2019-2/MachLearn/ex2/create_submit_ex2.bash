#!/bin/bash
rm -rf ex2_submit
mkdir ex2_submit
cp ex2.py ex2_submit/ex2.py
cp ex2_report.pdf ex2_submit/
echo "shahar siegman" > ex2_submit/details.txt
echo "011862141" >> ex2_submit/details.txt
zip ex2_shahar.zip ex2_submit/*

