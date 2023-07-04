# val=4 case
python3 src/render_wrap.py --checkpoints models/2023-06-29-12-51-13-dk86-n_121_a_30.00_r_0.80_2.40_indices-nerfacto/step-000030000.ckpt --cameras came
ra_n_121_a_30.00_r_0.80_2.40_val_4.json --no-use-crop

python3 src/render_wrap.py --checkpoints models/2023-06-29-12-51-13-dk86-n_121_a_30.00_r_0.80_2.40_indices-nerfacto/step-000030000.ckpt --cameras camera_crop_n_121_a_30.00_r_0.80_2.40_val_4.json --no-use-crop


# all wraps
python3 src/render_wrap.py --checkpoints models/2023*/step-000035000.ckpt --cameras camera_path_unbound.json

python3 src/render_wrap.py --checkpoints models/2023*/step-000035000.ckpt --cameras camera_path.json
