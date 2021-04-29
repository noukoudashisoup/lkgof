#!/bin/bash 

screen -AdmS ex2_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py betabinom_h0_dx10_n5_p1p11q11"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py betabinom_h0_dx10_n1_p1p11q11"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py betabinom_h0_dx10_n1_p1p11q11"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py lda_h1_dx100_v100_t3_p05q02"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py lda_h0_dx10_v5_t3_p05q1"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py lda_h1_dx10_v5_t3_p1q05"
screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py lda_h1_dx100_v100_t3_p1q05"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py lda_h0_dx100_v100_t3_p05q1"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "python ex2_vary_n_disc.py lda_h0_dx100_v5_t3_p05q06"
