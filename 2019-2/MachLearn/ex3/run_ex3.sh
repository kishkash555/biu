#! /bin/bash
if [ $# -eq 0 ]
  then
    echo "No arguments supplied"
else
    python ex3.py > $1 &
    sleep 3
    tail -f $1
    wait
fi