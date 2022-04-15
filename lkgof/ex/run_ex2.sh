#!/bin/bash 

screen -AdmS ex2_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h1_dx50_v10000_t3_p1q05temp1"
screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h1_dx50_v10000_t3_p1q05temp1e-1"
#screen -s ex2_kmod -x screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h1_dx50_v10000_t3_p1q08temp1e-1"
# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h0_dx50_v10000_t3_p05q06temp1e-1"
# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h0_dx50_v10000_t3_p05q06temp1"
# screen -S ex2_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h1_dx50_v10000_t3_p1q08temp1"
# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h1_dx50_v10000_t3_p1q05temp1"
# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_h0_dx50_v100_t3_p1q1temp1"

# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_mix_h1_dx50_v10000_t10_p5e-2q1e-2temp1e-1"
# screen -S ex2_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex2_vary_n_disc.py lda_mix_h1_dx50_v10000_t3_p05q02temp1e-1"