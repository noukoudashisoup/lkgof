#!/bin/bash 

screen -AdmS ex1_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h0_dx50_dz10"
# screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h0_dx50_dz10_p1_q11e-1"
# screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h1_dx50_dz10"
# screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h1_dx50_dz10_p3_q1"
# screen -S ex1_kmod -X screen -t tab2 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h1_dx50_dz10_p15e-1_q1"
 screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py isogdpm_h0_dx10_tr50_p1_q5"
# screen -S ex1_kmod -X screen -t tab2 bash -lic "conda activate lstein;python ex1_vary_n.py isogdpm_h1_dx10_tr10_p2_q1"
screen -S ex1_kmod -X screen -t tab2 bash -lic "conda activate lstein;python ex1_vary_n.py isogdpm_h1_dx10_tr50_p5_q1"




#screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h0_dx30_dz10"
#screen -S ex1_kmod -X screen -t tab1 bash -lic "conda activate lstein;python ex1_vary_n.py ppca_h1_dx30_dz10"
