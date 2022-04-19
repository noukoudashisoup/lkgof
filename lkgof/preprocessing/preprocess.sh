#!/bin/bash 

for CATEGORY in "stat.ME" "stat.AP" "stat.TH" "cs.LG" "math.PR"
do
    python trim.py -c $CATEGORY
    python makedocs.py -c $CATEGORY 
done
