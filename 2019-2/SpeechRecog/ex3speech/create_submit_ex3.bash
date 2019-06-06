#!/bin/bash
rm -rf ex3_submit
mkdir ex3_submit
cp *.py ex3_submit/
echo "shahar siegman" > ex3_submit/details.txt
echo "011862141" >> ex3_submit/details.txt
zip ex3_shahar.zip ex3_submit/*

