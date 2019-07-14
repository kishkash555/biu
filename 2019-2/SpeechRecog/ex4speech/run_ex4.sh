#! /bin/bash

if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    rev="$(git rev-parse HEAD)"
    logfile="results/result_log_${rev:0:6}_$1.txt"
    testfile="results/test_y_${rev:0:6}_$1.txt"
    echo "output is $logfile"
    echo "machine: $HOSTNAME pid: $!" > $logfile
    python ex4.py | tee >(grep -v ! >> $logfile) | sed -n -e s/!//p > $testfile &
    echo pid: $!
    sleep 3
    tail -f $logfile
 fi
