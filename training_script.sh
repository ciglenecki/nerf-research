#!/bin/bash
# nohup ./training_script.sh &

# Preprocess

# ns-process-data video --num-downscales 0 --data DIRORVID --output DIR
# ns-process-data video --num-downscales 0 --data data-raw/specimen_rgb.mp4 --output data/specimen_rgb

# ns-process-data video --num-downscales 0 --data data-raw/specimen_rgb.mp4 --output data/specimen_card_mask --colmap_feature_extractor_kwargs "\-\-ImageReader.mask_path data/specimen_card_mask/images"

# ns-process-data video --num-downscales 0 --data data-raw/specimen_rgb.mp4 --output data/specimen_hand_mask --colmap_feature_extractor_kwargs "\-\-ImageReader.mask_path data/specimen_hand_mask/images"


# Training

# data/i12_med_handheld
# data/i3_light
# data/i4_glare
# data/i4_images_dark
# data/v11_high_shakey_360
# data/v12_high_motion_360
# data/v3_low_motion
# data/v4_med_motion_vertical_handheld
# data/v5_med_motion
# data/v6_med_motion_shakey_handheld

# nohup /usr/bin/time --format='%C took %e seconds' \
# ns-train nerfacto --vis viewer \
# --logging.steps-per-log 2500 \
# --machine.num-gpus 1 \
# --viewer.quit-on-train-completion True \
# --data data/v12_high_motion_360 \
# > /root/projects/nerf-reserach/logs/log$(date +%F-%H-%M-%S).txt 2>&1 &

## declare an array variable


# Render only
ns-train nerfacto \
--data data/v11_high_shakey_360 \
--trainer.load-dir outputs/data-v11_high_shakey_360/nerfacto/2022-12-12_001633/nerfstudio_models \
--viewer.start-train False

--viewer.quit-on-train-completion True

function train() {
	echo "Running training"
	declare -a datasets=("v3_low_motion" "v5_med_motion" "v12_high_motion_360")
	declare -a models=("nerfacto")
	for model in "${models[@]}"; do
		for dataset in "${datasets[@]}"; do
			echo "Start dataset: $dataset $(date +%F-%H-%M-%S)";
			filepath_dataset="data/$dataset"
			logfile="/root/projects/nerf-reserach/logs/log-$dataset-$(date +%F-%H-%M-%S).txt"
			/usr/bin/time --format='%C took %e seconds' \
			ns-train $model \
			--vis viewer \
			--machine.num-gpus 1 \
			--viewer.quit-on-train-completion True \
			--logging.steps-per-log 100 \
			--data $filepath_dataset \
			2>&1 | tee $logfile
			wait;
			echo "End dataset: $dataset $(date +%F-%H-%M-%S)";
		done
	done
}

train