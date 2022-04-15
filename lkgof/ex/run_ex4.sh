#!/bin/bash 

screen -AdmS ex4_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py ppca_h0_dx100_dz10_p1_q1+1e-5"
# screen -S ex4_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py ppca_h1_dx100_dz10_p2_q1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py lda_h0_dx50_v1000_t3_p05mp1e-1"
# screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py lda_h0_dx50_v1000_t3_p0emp1"
# screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py lda_h1_dx50_v1000_t3_p1emp1"
# screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py lda_h0_dx50_v10000_t3_p05q1temp1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py lda_h1_dx50_v10000_t3_p1q05temp1e-1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex4_vary_mcsizes.py lda_h1_dx50_v10000_t3_p1q05temp1"
