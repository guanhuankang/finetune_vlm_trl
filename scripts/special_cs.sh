#!/bin/bash
#SBATCH --partition=special_cs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=5-00:00:00

cd /home/huankguan2/scratch/finetune_vlm_trl
mkdir -p log
source .venv/bin/activate

run_name=$(head -c2 /dev/urandom | od -An -tx1 | tr -d ' \n')-$(date +"%Y%m%d-%H%M%S")

python main.py --run_name "$run_name" "$@" > "log/${run_name}.out" 2> "log/${run_name}.err"
