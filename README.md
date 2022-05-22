# hsbcala

> train Calamari models for Upper Sorbian (Fraktur and Antiqua) prints on HPC

Scripts for training [Calamari OCR](https://github.com/Calamari-OCR/calamari) models on [ZIH's Power9 NVidia V100 HPC cluster](https://doc.zih.tu-dresden.de/jobs_and_resources/hardware_overview/#ibm-power9-nodes-for-machine-learning) for [Upper Sorbian](https://www.sorabicon.de/en/home/) prints.

The GT data is [here for Fraktur](https://mrocel.sorbib.de/index.php/s/XstEfxREcf7LQEj) and [here for Antiqua](https://mrocel.sorbib.de/index.php/s/emcjiHz3MZFZtdW). Production and rights: [Sorbian Institute](https://www.serbski-institut.de/en).

The approach was to do **finetuning** on pretrained models:
- for Fraktur prints:
  - with Calamari 2: [deep3_fraktur19](https://github.com/Calamari-OCR/calamari_models_experimental)
  - with Calamari 1: [fraktur_19th_century](https://github.com/Calamari-OCR/calamari_models)
- for Antiqua prints:
  - with Calamari 2: [deep3_lsh4](https://github.com/Calamari-OCR/calamari_models_experimental)
  - with Calamari 1: [antiqua_historical](https://github.com/Calamari-OCR/calamari_models)

(We don't want to use voting during inference, therefore we use `calamari-train` – not `calamari-cross-fold-train` – and pick the first model among the pretrained ensembles, respectively. We use Calamari 2.2.2 / Calamari 1.0.5 CLIs – in an attempt to find similar settings for both versions.)

This repo provides the [Slurm](https://doc.zih.tu-dresden.de/jobs_and_resources/slurm/) scripts, which:
1. source an environment script `ocrenv.sh` loading the HPC environment's [modules](https://doc.zih.tu-dresden.de/software/modules/) (an [Lmod](https://www.tacc.utexas.edu/research-development/tacc-projects/lmod) system) and a custom venv (`powerai-kernel2.txt`)
2. checks whether any checkpoints exist in the output directory already –
   - if yes, then use `calamari-resume-training`
   - otherwise, start `calamari-train`
3. sets up all parameters
4. wraps the call with [Nvidia Nsight](https://developer.nvidia.com/nsight-systems) for profiling

For optimal resource allocation (empirically determined via Nsight and the [PIKA system](https://doc.zih.tu-dresden.de/software/pika/) for job monitoring), we use
- a large batch size (64-80)
- a large number (10) of cores and data workers
- a high amount of RAM (32 GB) per core, with full preloading and data prefetching (32)

For optimal accuracy, we use
- re-computing the codec (i.e. keeping only shared codepoints, adding new ones)
- implicit augmentation (5-fold)
- early stopping (at 10 epochs without improvement)

## Results

...see release archives for model files

## Evaluation

...on held out validation data (used for checkpoint selection):

| **model** | **CER** |
| --- | --- |
| hsbfraktur.cala1 | 1.4% |
| hsbfraktur.cala | 0.7% |
| hsblatin.cala1 | 1.3% |
| hsblatin.cala | 0.6% |

...on truly representative extra data: to come
