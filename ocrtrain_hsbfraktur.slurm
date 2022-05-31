#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --partition=ml # Power9/ppc64le
#SBATCH --gres=gpu:4 # with 4 GPUs (must be specified here)
#SBATCH --time=10:00:00   # walltime
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1  # limit to one node
#SBATCH --cpus-per-task=21  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1443M # memory per core
#SBATCH -J ocrtrain_hsbfraktur   # job name
#SBATCH -A p_da_layout
#SBATCH --chdir /projects/p_da_layout/ocrbeegws

# todo: chain job every 5 or so checkpoints
# todo: ressource optimization (GPU RAM, CPU RAM, CPUs, nodes, workers)

set -e
shopt -s nullglob
source /projects/p_da_layout/ocrenv.sh

tar -zxvf /projects/p_da_layout/ocrwaws/hsbfraktur2.tar.gz -C /dev/shm/
mkdir -p hsbfraktur.cala

# too many files per dir → untar to /dev/shm instead of loading from FS
# /dev/shm already in RAM → disable preloading from disk 
# multi-GPU → dist_strategy=mirror instead of default
# prefetch larger than batch_size
cmd=(calamari-train \
 --trainer.output_dir hsbfraktur.cala \
 --train.images "/dev/shm/hsbfraktur.train/*.png" \
 --val.images "/dev/shm/hsbfraktur.val/*.png" \
 --train.preload false \
 --val.preload false \
 --train.skip_invalid true \
 --val.skip_invalid true \
 --train.num_processes 10 \
 --val.num_processes 10 \
 --train.prefetch 128 \
 --val.prefetch 128 \
 --network deep3 \
 --warmstart.model calamari_models_experimental/deep3_fraktur19/0.ckpt.json \
 --codec.keep_loaded False \
 --train.batch_size 64 \
 --val.batch_size 64 \
 --trainer.device.dist_strategy mirror \
 --trainer.random_seed 50 \
 --early_stopping.n_to_go 10 \
 --trainer.progress_bar_mode 0 \
 --n_augmentations 5 \
 --trainer.epochs 500)
checkpoints=(hsbfraktur.cala/checkpoint/checkpoint_*/trainer_params.json)
if ((${#checkpoints[*]})); then
    cmd=(calamari-resume-training ${checkpoints[-1]})
fi

echo "${cmd[@]}"
nsys profile --force-overwrite true -o hsbfraktur.cala/qdstrm.log "${cmd[@]}"

# --device.gpus 0 \
# --trainer.gen SplitTrain --trainer.gen.validation_split_ratio=0.2 \
# --data.line_height 64 # default: 48
# --train.batch_size 32 # default: 16
# --early_stopping.n_to_go 5 # default: -1 (i.e. no early stopping)
# --trainer.epochs 500 # default: 100
# --n_augmentations 5 # default: 0
# calamari-cross-fold-train
# --train.preload False --val.preload False
# --early_stopping.frequency 5 # default: only once per epoch
# --network deep3
# --network htr+ --data.line_height 64
