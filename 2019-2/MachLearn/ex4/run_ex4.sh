#! /bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    rev="$(git rev-parse HEAD)"
    rev6="results/result_log_${ot:0:6}_$1.txt"
    python ex4.py > $rev6 &
    sleep 3
    tail -f $rev6
    wait
fi
