#!/bin/bash
#SBATCH -N 1 # number of nodes
#SBATCH -p cox
#SBATCH -n 8 # number of cores
#SBATCH --mem 10000 # memory pool for all cores
#SBATCH --gres=gpu:1 # memory pool for all cores
#SBATCH -t 1-00:00 # time (D-HH:MM)

#SBATCH -o slurm_train/_train_%j.%N.out # STDOUT
#SBATCH -e slurm_train/_train_%j.%N.err # STDERR

module load python/3.6.3-fasrc01
module load cuda/9.2.88-fasrc01
source activate pytorch-vae
module load GCCcore/6.4.0
nvidia-smi -L

# Run (-u to prevent buffering)
# python run.py -c configs/vae.yaml #MNIST
python run.py -c configs/vae_mvtec.yaml #MVTec VAE
# python run.py -c configs/ae_mvtec.yaml #MVTec AE
# python run.py -c configs/"$1" #MVTec objects