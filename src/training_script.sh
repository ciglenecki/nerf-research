#!/bin/sh

# nohup COMMAND >/dev/null 2>&1 &

# function run() {
# 	echo "Running training"
# 	declare -a datasets=("v3_low_motion" "v5_med_motion" "v12_high_motion_360")
# 	declare -a models=("nerfacto")
# 	for model in "${models[@]}"; do
# 		for dataset in "${datasets[@]}"; do
# 			echo "Start dataset: $dataset $(date +%F-%H-%M-%S)";
# 			filepath_dataset="data/$dataset"
# 			logfile="/root/projects/nerf-reserach/logs/log-$dataset-$(date +%F-%H-%M-%S).txt"
# 			/usr/bin/time --format='%C took %e seconds' \
# 			ns-train $model \
# 			--vis viewer \
# 			--machine.num-gpus 1 \
# 			--viewer.quit-on-train-completion True \
# 			--logging.steps-per-log 100 \
# 			--data $filepath_dataset \
# 			2>&1 | tee $logfile
# 			wait;
# 			echo "End dataset: $dataset $(date +%F-%H-%M-%S)";
# 		done
# 	done
# }

function run {
	python3 nerfstudio/scripts/train_wrap.py --models nerfacto --max-num-iterations 100_000 --datasets data/specimen_rgb_n_02 data/specimen_rgb_n_03 data/specimen_rgb_n_04 data/specimen_rgb_n_05 data/specimen_rgb_n_06 data/specimen_rgb_n_07 data/specimen_rgb_n_08 data/specimen_rgb_n_12 data/specimen_rgb_n_18 data/specimen_rgb_n_24 data/specimen_rgb_n_30 data/specimen_rgb_n_36 data/specimen_rgb_n_42 data/specimen_rgb_n_48 data/specimen_rgb_n_54;
	echo "py"
}

run