#!/bin/bash 

screen -AdmS ex3_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py isogdpm_ms_dx10_tr10_q1"
screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py isogdpm_ms_dx10_tr5_q1"

# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py ppca_ws_dx50_h0_p1"
# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_as_dx50_h0_p05"

# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_mix_dx50_h1_q1e-2temp1e-1alpha1e-2"
# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_mix_dx50_h1_q1e-2temp1e-1alpha1e-1"
# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_mix_dx50_h1_q1e-2temp1e-1alpha1"
# 
# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_mix_dx50_h1_q1e-2temp1alpha1e-2"
# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_mix_dx50_h1_q1e-2temp1alpha1e-1"
# screen -S ex3_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex3_prob_params.py lda_mix_dx50_h1_q1e-2temp1alpha1"
