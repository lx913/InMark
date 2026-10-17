#!/bin/bash

# Submit this script with: sbatch test_distortion.slurm

#SBATCH --job-name=mscoco_test_distortion
#SBATCH --output=/scratch/%u/%x-%j-%N.out
#SBATCH --error=/scratch/%u/%x-%j-%N.err
#SBATCH --time=0-24:00:00
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --partition contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --array=8
#SBATCH --mail-user=xli62@gmu.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load miniconda3
source ~/.bashrc
conda init bash
conda activate xli62

distortion_types=(
  color_jitter
  crop
  blur
  noise
  jpeg_compress
  resize
  sharpness
  SDEdit
  SDEdit2
)

num_distortions=${#distortion_types[@]}

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "SLURM_ARRAY_TASK_ID is not set. Please submit with sbatch."
  exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "$num_distortions" ]; then
  echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (max: $((num_distortions - 1)))."
  exit 1
fi

distortion_type=${distortion_types[$SLURM_ARRAY_TASK_ID]}

echo "Running distortion config ${SLURM_ARRAY_TASK_ID}/${num_distortions}: ${distortion_type}"

python test_distortion.py \
  --distortion_type "${distortion_type}"
