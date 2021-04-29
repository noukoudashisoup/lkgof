#!/bin/bash 

screen -AdmS ex4_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h0_dx10_dz5_p1q2"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h1_dx10_dz5_p2q1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h0_dx50_dz10_p1q2"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h1_dx50_dz10_p2q1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h1_dx50_dz10_p2q5e-1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h1_dx100_dz10_p2q1"

screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h1_dx100_dz10_p1q5e-1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h0_dx100_dz10_p5e-1q1"
#screen -S ex4_kmod -X screen -t tab1 bash -lic "python ex4_vary_n.py ppca_h0_dx100_dz10_p1_same"
