export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export MODEL_NAME="Manojb/stable-diffusion-2-1-base"
versions=('v1')
methods=("gt_dm_watermark")
length=${#versions[@]}
lr=1e-4
snr_gamma=1.0


for method in "${methods[@]}"; do
  for ((i=0; i<length; i++)); do
    version="${versions[$i]}"
    export OUTPUT_DIR="./UNet/checkpoint/${method}_${version}/all"
    export DATASET_NAME="../dataset/MSCOCO2017/"
    export LOG_DIR="./UNet/train/logs"
    export CACHE_DIR="../dataset/MSCOCO2017"
    echo "Current OUTPUT_DIR: $OUTPUT_DIR, $i"

    if [ "$method" = "gt_dm_watermark" ]; then
      echo "script execution. ${method}"
      accelerate launch --mixed_precision="bf16" watermarking_diffusion_model.py \
        --pretrained_model_name_or_path="$MODEL_NAME" \
        --dataset_name=mscoco --cache_dir "$CACHE_DIR" \
        --train_data_dir "$DATASET_NAME" --image_column='img' --caption_column="label" \
        --report_to=tensorboard \
        --resolution=512 --random_flip \
        --train_batch_size=32 \
        --num_train_epochs=50 --checkpointing_steps=10000 \
        --learning_rate "$lr" --lr_scheduler="constant" \
        --seed=42 \
        --output_dir="$OUTPUT_DIR" \
        --guidance_token=8 \
        --logging_dir "$LOG_DIR" \
        --exp_id "$i" \
        $snr_arg
      echo "All processes completed"
    else
      echo "Method not implemented: $method" >&2
      exit 1
    fi
  done
done
