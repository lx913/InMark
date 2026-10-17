#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --job-name=mscoco_ourmethod_lr1e-4_snr1.0  ##name that will show up in the queue
#SBATCH --output=/scratch/%u/%x-%j-%N.out   ##filename of the output
#SBATCH --error=/scratch/%u/%x-%j-%N.err   ##filename of the error output
#SBATCH --time=0-24:00:00  ## Job 的最长运行时间，格式 day-hour:min:sec
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16  ## CPU核心数
#SBATCH --mem=32GB  ## 内存
#SBATCH --partition contrib-gpuq
#SBATCH --gres=gpu:A100.80gb:4  ##GPU(s) 规格和数量

export CUDA_VISIBLE_DEVICES=0,1,2,3
#SBATCH --mail-user xli62@gmu.edu  ##your email address
#SBATCH --mail-type END  ##slurm will email you when your job ends
#SBATCH --mail-type FAIL  ##slurm will email you when your job fails

## Load modules, insert code, and run your programs here.
module load gnu10/10.3.0-ya
source ~/.bashrc
conda init bash
conda activate xli62

export MODEL_NAME="Manojb/stable-diffusion-2-1-base"
versions=('v1')
methods=("gt_dm_watermark")
length=${#versions[@]}
lr=1e-4
snr_gamma=1.0

snr_arg=""
if [ "$snr_gamma" != "none" ]; then
  snr_arg="--snr_gamma=${snr_gamma}"
fi

# Loop through the array and set the OUTPUT_DIR variable accordingly
for method in "${methods[@]}"; do
  for ((i=0; i<$length; i++)); do 
      version="${versions[$i]}"
      export OUTPUT_DIR="./LoRA/checkpoint/${method}_lr${lr}_snr${snr_gamma}_${version}/all"
      export DATASET_NAME="../dataset/MSCOCO2017/"
      export LOG_DIR="./LoRA/train/logs"
      export CACHE_DIR="../dataset/MSCOCO2017"
      echo "Current OUTPUT_DIR: $OUTPUT_DIR, $i"
      if [ -f "$OUTPUT_DIR/pytorch_lora_weights.bin" ] || [ -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]; then
        echo "Folder exists. Skipping script execution."
      else
        if [ "$method" == "gt_dm_watermark" ]; then
          echo "script execution. ${method}"
          accelerate launch --mixed_precision="bf16" ./watermark_diffusion_model.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --dataset_name=mscoco --cache_dir $CACHE_DIR\
          --train_data_dir $DATASET_NAME --image_column='img' --caption_column="label" \
          --report_to=tensorboard \
          --resolution=512 --random_flip \
          --train_batch_size=32 \
          --num_train_epochs=50 --checkpointing_steps=10000 \
          --learning_rate ${lr} --lr_scheduler="constant" \
          --seed=42 \
          --output_dir=${OUTPUT_DIR} \
          --guidance_token=8 \
          --logging_dir $LOG_DIR \
          --exp_id ${i} \
          ${snr_arg}
        else
          echo "Method not implemented"
        fi
        wait
        echo "All processes completed"
      fi
  done
done
