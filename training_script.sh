#!/bin/bash
# nohup ./training_script.sh &

# Preprocess


# ns-process-data images --data data-raw/i12_med_handheld --output data/i12_med_handheld --num-downscales 0
# ns-process-data images --data data-raw/i4_glare --output data/i4_glare --num-downscales 0
# ns-process-data images --data data-raw/i4_images_dark --output data/i4_images_dark --num-downscales 0
# ns-process-data video --data data-raw/v12_high_motion_360.mp4 --output data/v12_high_motion_360 --num-downscales 0
# ns-process-data video --data data-raw/v3_low_motion.mp4 --output data/v3_low_motion --num-downscales 0
# ns-process-data video --data data-raw/v4_med_motion_vertical_handheld.mp4 --output data/v4_med_motion_vertical_handheld --num-downscales 0
# ns-process-data video --data data-raw/v5_med_motion.mp4 --output data/v5_med_motion --num-downscales 0
# ns-process-data video --data data-raw/v6_med_motion_shakey_handheld.mp4 --output data/v6_med_motion_shakey_handheld --num-downscales 0


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