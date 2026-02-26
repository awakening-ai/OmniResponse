for s in 0 1 2 3 4; do
  python frechet_video_distance.py  \
    --dir_a /path/to//test/gen-listener-video  --dir_b /path/to/listener_224_cv  \
    --num_videos 140 \
    --num_frames 256 \
    --batch_size 32 \
    --seed $((42+s)) \
    --detector_path /path/to/i3d_torchscript.pt
done