#!/bin/bash 

python train_lda.py -c "stat.ME" "math.PR" "stat.TH" 
python testdata.py -c "stat.ME" "math.PR" "stat.TH" -t "stat.TH"

# python train_lda.py -c "cs.LG" "stat.ME" "stat.TH" 
# python testdata.py -c "cs.LG" "stat.ME" "stat.TH" -t "stat.TH"

