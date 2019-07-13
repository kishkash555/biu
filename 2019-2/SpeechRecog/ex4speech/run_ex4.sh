#! /bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    rev="$(git rev-parse HEAD)"
    rev6="results/result_log_${rev:0:6}_$1.txt"
    echo "output is $rev6"
    python ex4.py > $rev6 &
    echo pid: $!
    sleep 3
    tail -f $rev6
 fi
