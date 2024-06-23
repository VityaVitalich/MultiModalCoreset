#!/bin/bash

#SBATCH --job-name=llmcompr

#SBATCH --partition=ais-gpu

#SBATCH --mail-type=ALL

#SBATCH --mail-user=V.Moskvoretskii@skoltech.ru

#SBATCH --output=zh_logs/dist.txt
#SBATCH --time=0-05

#SBATCH --mem=256G

#SBATCH --nodes=1

#SBATCH -c 8

#SBATCH --gpus=1

srun singularity exec --bind /trinity/home/v.moskvoretskii/:/home -f --nv /trinity/home/v.moskvoretskii/images/multimae.sif bash -c '
    ls;
    cd /home;
    export HF_TOKEN=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;
    export SAVING_DIR=/home/cache/;
    export WANDB_API_KEY=2b740bffb4c588c7274a6e8cf4e39bd56344d492;
    cd /home/MultiModalCoreset/Dataset_Quantization/coreset;
    nvidia-smi;
    pip list;
    CUDA_LAUNCH_BLOCKING=1;
    python calc_dist.py;
'

