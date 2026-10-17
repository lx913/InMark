model_path="UNet/checkpoint/gt_dm_watermark_v1/all/unet_final.pt"
guidance=7.5
steps=30
scheduler="dpms_m"
vae="original"

python generate_text2img.py \
  -model_path "$model_path" \
  -guidance "$guidance" \
  -steps "$steps" \
  -scheduler "$scheduler" \
  -vae "$vae" \
  "$@"
