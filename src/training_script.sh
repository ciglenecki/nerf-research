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