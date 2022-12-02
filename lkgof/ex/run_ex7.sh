#!/bin/bash 

screen -AdmS ex7_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

# screen -S ex7_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex7_prob_params_heat.py ppca_ws_dx100_dz10"
screen -S ex7_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex7_prob_params_heat.py lda_as_dx50_v10000_t3_temp1"

