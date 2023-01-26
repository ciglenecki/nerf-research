#!/bin/sh

# nohup COMMAND >/dev/null 2>&1 &

function run {
	echo "Running $(date +%F-%H-%M-%S)";
	logfile="/root/projects/nerf-reserach/logs/log.txt"
	/usr/bin/time --format='%C took %e seconds' \
	python3 src/eval_checkpoints.py --checkpoints models.txt 2>&1 | tee log.txt
	echo "End $(date +%F-%H-%M-%S)";
}

run