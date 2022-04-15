#!/bin/bash 

screen -AdmS ex6_kmod -t tab0 bash 
# launch each problem in parallell, each in its own screen tab
# See http://unix.stackexchange.com/questions/74785/how-to-open-tabs-windows-in-gnu-screen-execute-commands-within-each-one
# http://stackoverflow.com/questions/7120426/invoke-bash-run-commands-inside-new-shell-then-give-control-back-to-user

#screen -S ex6_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_AP_ME_TH"
# screen -S ex6_kmod -X screen -t tab1 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_mathPR_statME_statTH"
screen -S ex6_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_csLG_ME_TH"
# screen -S ex6_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_csLG_ML_TH"
# screen -S ex6_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_cond-mat.mtrl-sci_cond-mat.mes-hall"
# screen -S ex6_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_hep-th_quant-ph_P06_Q07_Rquant-ph"
# screen -S ex6_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_hep-ph_quant-ph_P06_Q07_Rhep-ph"
# screen -S ex6_kmod -X screen -t tab2 bash -lic "conda activate lkgof;python ex6_real_data.py arxiv_astro-ph.GA_astro-ph.SR_P06_Q07_Rastro-ph.SR"



