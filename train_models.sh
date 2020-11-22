#!/bin/bash
#SBATCH -o /home/jiaming/scirdb/Logs/slurm.%j.out
#SBATCH -e /home/jiaming/scirdb/Logs/slurm.%j.err
#SBATCH --mail-user=jiaming@stanford.edu
#SBATCH --mail-type=END,FAIL
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH -t 7-00:00:00

echo ${model}
echo ${alg}
echo ${vs}
echo ${w}

cd /home/jiaming/scirdb/code/treatment_identification

module load anaconda
source activate /share/pi/rubin/jiaming/envs/rjupyter

#python3 -u evaluate_embeddings.py -m ${model} -vs ${vs} -a ${alg} -w ${w} -ct prostate -c 70
python3 -u evaluate_embeddings.py -m ${model} -vs ${vs} -a ${alg} -w ${w} -ct oropharynx -c 40
#python3 -u evaluate_embeddings.py -m ${model} -vs ${vs} -a ${alg} -w ${w} -ct esophagus -c 50
#python3 -u evaluate_struc_bow.py -ct prostate -c 70
#python3 -u evaluate_struc_bow.py -ct oropharynx -c 40
#python3 -u evaluate_struc_bow.py -ct esophagus -c 50
