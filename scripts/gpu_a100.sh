#!/bin/bash
run_name=$(date +"%Y%m%d-%H%M%S")-$(head -c2 /dev/urandom | od -An -tx1 | tr -d ' \n')

args="$@"

working_dir=log/${run_name}/

mkdir -p $working_dir

cat > ${working_dir}/job_${run_name}.slurm <<EOL
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256GB
#SBATCH --time=5-00:00:00
#SBATCH --output=log/${run_name}.out
#SBATCH --error=log/${run_name}.err

cd /home/huankguan2/scratch/finetune_vlm_trl
source .venv/bin/activate

python main.py --run_name ${run_name} ${args}
EOL

# Submit job
cd ${working_dir}
sbatch job_${run_name}.slurm
