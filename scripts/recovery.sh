#!/bin/bash
com_str="cd .. ; "

for var in "$@"
do
    com_str+="julia --project=. -p 128 ./scripts/recovery_analysis_fit_final.jl $var ; "
done

com_str+="exit"
eval $com_str