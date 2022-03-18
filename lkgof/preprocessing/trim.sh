#!/bin/bash 

for CATEGORY in "stat.ME" "stat.AP" "stat.TH" "cs.LG" "math.PR"
#for CATEGORY in "math.PR"
do
    # python trim.py -c $CATEGORY
    python makedocs.py -c $CATEGORY
done
