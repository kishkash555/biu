#! /bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    rev="$(git rev-parse HEAD)"
    logfile="results/result_log_${rev:0:6}_$1.txt"
    testfile="results/test_y_${rev:0:6}_$1.txt"
    echo "output is $logfile"
    python ex4.py $testfile >> $logfile &
    echo "machine: $HOSTNAME pid: $!" >> $logfile
    sleep 3
    tail -f $logfile
 fi
