#!/bin/bash
#SBATCH --job-name=ViLT     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=24:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=jmonas@princeton.edu
#SBATCH --partition=mig

module purge
module load cudatoolkit/11.1
module load cudnn/cuda-11.x/8.2.0
module load anaconda3/2023.9
conda activate torch-env

python eval.py --validate