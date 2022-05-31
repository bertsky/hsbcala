#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --partition=ml # Power9/ppc64le
#SBATCH --gres=gpu:1 # with 1 GPUs (must be specified here)
#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1  # limit to one node
#SBATCH --cpus-per-task=22  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1443M # memory per core
#SBATCH -J ocrtrain1_hsbfraktur   # job name
#SBATCH -A p_da_layout
#SBATCH --chdir /projects/p_da_layout/ocrbeegws

# todo: chain job every 5 or so checkpoints
# todo: ressource optimization (GPU RAM, CPU RAM, CPUs, nodes, workers)

set -e
shopt -s nullglob
source /projects/p_da_layout/ocrenv1.sh

tar -zxvf /projects/p_da_layout/ocrwaws/hsbfraktur2.tar.gz -C /dev/shm/
mkdir -p hsbfraktur.cala1

cmd=(calamari-train \
 --output_dir hsbfraktur.cala1 \
 --files "/dev/shm/hsbfraktur.train/*.png" \
 --validation "/dev/shm/hsbfraktur.val/*.png" \
 --train_data_on_the_fly \
 --validation_data_on_the_fly \
 --num_threads 10 \
 --weights calamari_models/fraktur_19th_century/0.ckpt.json \
 --seed 50 \
 --batch_size=80 \
 --max_iters 5000000 \
 --early_stopping_frequency 1.0 \
 --early_stopping_nbest 10 \
 --n_augmentations 5 \
 --no_progress_bars \
 --display 0.1)
checkpoints=(hsbfraktur.cala1/model_*.ckpt.json)
if ((${#checkpoints[*]})); then
    cmd=(calamari-resume-training --checkpoint ${checkpoints[-1]%.json} \
	--files "/dev/shm/hsbfraktur.train/*.png" \
	--validation "/dev/shm/hsbfraktur.val/*.png")
fi

echo "${cmd[@]}"
nsys profile --force-overwrite true -o hsbfraktur.cala1/qdstrm.log "${cmd[@]}" 2>&1 | tee hsbfraktur.cala1/train.log

# --text_regularization ?
# --line_height 64 # default: 48
# --network "cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5" \
# --batch_size 32 # default: 16
# --max_iters 5000000 # default: 1000000
# --n_augmentations 5 # default: 0
# calamari-cross-fold-train
# --train_data_on_the_fly --validation_data_on_the_fly
# --early_stopping_frequency 5 # default: only once per epoch
# --keep_loaded_codec \

