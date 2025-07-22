#!/bin/bash
#SBATCH --partition=special_cs
#SBATCH --nodes=1                # 1 computer nodes
#SBATCH --ntasks-per-node=1      # 1 MPI tasks on EACH NODE
#SBATCH --cpus-per-task=8        # 4 OpenMP threads on EACH MPI TASK
#SBATCH --gres=gpu:a100:1        # Using 1 GPU card
#SBATCH --mem=512GB               # Request 50GB memory
#SBATCH --time=5-00:00:00        # Time limit day-hrs:min:sec
#SBATCH --output=log/%j/output.txt   # Standard output
#SBATCH --error=log/%j/error.txt    # Standard error log

### Shell Here ###
cd /home/huankguan2/scratch/finetune_vlm_trl
source .venv/bin/activate
python main.py --quick_eval
