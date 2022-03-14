#!/bin/bash 

screen -AdmS ex5_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex5_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex5_kernel_params.py ppca_h0_dx100_dz10"
# screen -S ex5_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex5_kernel_params.py ppca_h1_dx100_dz10_p2_q1"
# screen -S ex5_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex5_kernel_params.py lda_h1_dx50_v1000_t3_p1q05temp1"
#screen -S ex5_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex5_kernel_params.py lda_h1_dx50_v10000_t3_p1q05temp1"


