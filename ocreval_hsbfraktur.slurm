#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --partition=ml # Power9/ppc64le
#SBATCH --gres=gpu:1 # with 1 GPUs (must be specified here)
#SBATCH --time=1:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1  # limit to one node
#SBATCH --cpus-per-task=21  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1443M # memory per core
#SBATCH -J ocreval_hsbfraktur   # job name
#SBATCH -A p_da_layout
#SBATCH --chdir /projects/p_da_layout/ocrbeegws

set -e
shopt -s nullglob
source /projects/p_da_layout/ocrenv.sh

tar -zxvf /projects/p_da_layout/ocrwaws/hsbfraktur2.tar.gz -C /dev/shm/
test -d hsbfraktur.cala
test -f hsbfraktur.cala/best.ckpt.json

cmd=(calamari-predict \
 --checkpoint hsbfraktur.cala/best.ckpt.json \
 --data.preload false \
 --pipeline.num_processes 10 \
 --pipeline.batch_size 64 \
 --data.images "/dev/shm/hsbfraktur.val/*.png" \
 --predictor.progress_bar false)
echo "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee hsbfraktur.cala/eval.log

cmd=(calamari-eval \
 --checkpoint hsbfraktur.cala/best.ckpt.json \
 --gt.preload false \
 --gt.texts "/dev/shm/hsbfraktur.val/*.gt.txt" \
 --evaluator.progress_bar false)
echo "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee -a hsbfraktur.cala/eval.log

cmd=(calamari-eval \
 --checkpoint hsbfraktur.cala/best.ckpt.json \
 --gt.preload false \
 --gt.texts "/dev/shm/hsbfraktur.val/*_GTOLENA*.gt.txt" \
 --gt.texts "/dev/shm/hsbfraktur.val/*_GTSBB*.gt.txt" \
 --gt.texts "/dev/shm/hsbfraktur.val/*_GTOCROPY*.gt.txt" \
 --gt.texts "/dev/shm/hsbfraktur.val/*_GTLINES*.gt.txt" \
 --evaluator.progress_bar false)
echo "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee hsbfraktur.cala/eval.bin.log

cmd=(calamari-eval \
 --checkpoint hsbfraktur.cala/best.ckpt.json \
 --gt.preload false \
 --gt.texts "/dev/shm/hsbfraktur.val/*_GTPAGE*.gt.txt" \
 --evaluator.progress_bar false)
echo "${cmd[@]}"
"${cmd[@]}" 2>&1 | tee hsbfraktur.cala/eval.raw.log
