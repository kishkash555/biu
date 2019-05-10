#!/bin/bash
rm -rf ex2_submit
mkdir ex2_submit
cp speech_ex2.py ex2_submit/ex2.py
echo "shahar siegman" > ex2_submit/details.txt
echo "011862141" >> ex2_submit/details.txt
cp ex2_0426cf_out.txt ex2_submit/output.txt
zip ex2_shahar.zip ex2_submit/*

