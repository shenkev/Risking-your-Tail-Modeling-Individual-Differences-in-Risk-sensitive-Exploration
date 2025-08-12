#!/bin/bash
com_str="cd .. ; "

for var in "$@"
do
    com_str+="julia --project=. -p 128 ./scripts/fit_abcsmc_context_final.jl $var ; "
done

com_str+="exit"
eval $com_str