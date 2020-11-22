#!/bin/bash

#simulation submission
models=("fasttext")
algs=(0)
vector_size=(300)
window=(5)

#alpha=(0.0025 0.025 0.25)
#sample=(1e-4 1e-2 0)

for model in ${models[*]} 
do
    for alg in ${algs[*]} 
    do
        for vs in ${vector_size[*]} 
        do
            #for a in ${alpha[*]}
            #do
            for w in ${window[*]}
            do
                sbatch --export=model=${model},alg=${alg},vs=${vs},w=${w} train_models.sh
            done
            #done
        done
    done 
done
