#!/bin/sh

# nohup COMMAND >/dev/null 2>&1 &

function run {
	timestamp=$(date +%F-%H-%M-%S)
	echo "Running $timestamp";
	logfile="/root/projects/nerf-reserach/logs/log-$timestamp.txt"
	/usr/bin/time --format='%C took %e seconds' \
	python3 nerfstudio/scripts/train_wrap.py --models nerfacto --max-num-iterations 30_000 --datasets data/specimen_rgb/ \
	2>&1 | tee log.txt
	echo "End $(date +%F-%H-%M-%S)";
}

run