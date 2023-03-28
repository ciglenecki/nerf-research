#!/bin/bash

# nohup src/run.sh >/dev/null 2>&1 &

function run {
	timestamp=$(date +%F-%H-%M-%S);
	logfile="/root/projects/nerf-research/logs/log-$timestamp.txt";
	touch $logfile;
	echo "Logging to $logfile";
	echo "Running $timestamp" >> $logfile;
	/usr/bin/time --format='%C took %e seconds' ./src/colmaps.sh 2>&1 | tee $logfile
	echo "End $(date +%F-%H-%M-%S)" >> $logfile;
}

run