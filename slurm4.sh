#!/bin/bash
#SBATCH --partition=special_cs
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=32        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:a100:4        # Using 1 GPU card
#SBATCH --mem=1024GB               # Request 50GB memory
#SBATCH --time=1-06:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=log/%j/output_GPUx4.txt   # Standard output
#SBATCH --error=log/%j/error_GPUx4.txt    # Standard error log

### Shell Here ###
cd /home/huankguan2/scratch/finetune_vlm_trl
source .venv/bin/activate
python main.py --num_gpus 4 --gradient_accumulation_steps 1
