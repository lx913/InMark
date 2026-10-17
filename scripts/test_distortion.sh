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

for i in "${!distortion_types[@]}"; do
  distortion_type="${distortion_types[$i]}"
  echo "Running distortion $((i + 1))/${#distortion_types[@]}: ${distortion_type}"
  python test_distortion.py --distortion_type "$distortion_type" "$@"
done
