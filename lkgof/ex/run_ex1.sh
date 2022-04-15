#!/bin/bash 

screen -AdmS ex1_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

# screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex1_vary_n.py ppca_h0_dx50_dz10_p1_q1"
# screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex1_vary_n.py ppca_h0_dx100_dz10_p1_q1+1e-5"
# screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex1_vary_n.py ppca_h1_dx100_dz10"
screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex1_vary_n.py ppca_h1_dx100_dz10_p3_q1"
