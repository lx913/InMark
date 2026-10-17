#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --job-name=mscoco_generate_text2img  ##name that will show up in the queue
#SBATCH --output=/scratch/%u/%x-%j-%N.out   ##filename of the output
#SBATCH --error=/scratch/%u/%x-%j-%N.err   ##filename of the error output
#SBATCH --time=0-24:00:00  ## Job 的最长运行时间，格式 day-hour:min:sec
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8  ## CPU核心数
#SBATCH --mem=8GB  ## 内存
#SBATCH --partition contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1  ##GPU(s) 规格和数量
#SBATCH --array=0

# export CUDA_VISIBLE_DEVICES=0,1,2,3
#SBATCH --mail-user xli62@gmu.edu  ##your email address
#SBATCH --mail-type END  ##slurm will email you when your job ends
#SBATCH --mail-type FAIL  ##slurm will email you when your job fails

## Load modules, insert code, and run your programs here.
module load miniconda3
source ~/.bashrc
conda init bash
conda activate xli62

epoch=50
# lr_values=(1e-3 5e-4 1e-4 5e-5 1e-5 5e-6)
lr_values=(1e-4)
snr_gamma_values=(1.0)
# snr_gamma_values=(none 5.0 3.0 1.0 0.25)

num_lr=${#lr_values[@]}
num_snr=${#snr_gamma_values[@]}
total_configs=$((num_lr * num_snr))

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Please submit with sbatch."
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_configs" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (max: $((total_configs - 1)))."
  exit 1
fi

config_id=$SLURM_ARRAY_TASK_ID
lr_idx=$(( config_id / num_snr ))
snr_idx=$(( config_id % num_snr ))

lr=${lr_values[$lr_idx]}
snr_gamma=${snr_gamma_values[$snr_idx]}
snr_arg="-snr_gamma ${snr_gamma}"

echo "Grid config ${config_id}/${total_configs}: lr=${lr}, snr_gamma=${snr_gamma}"

python /scratch/xli62/Diff_MSCOCO_Watermark_withdataaug/generate_text2img.py -learning_rate ${lr} ${snr_arg} -epoch ${epoch}
