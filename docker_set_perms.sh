#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Required arguments for setting the permission were not provdied. Permissions will not be changed."
else
	USER_ID=$1
	USER_NAME=$2
	GROUP_ID=$3
	GROUP_NAME=$4

	echo "Setting permissions to $USER_NAME:$GROUP_NAME"
	# ====== START PERMISSION SETTING
	addgroup --gid $GROUP_ID $GROUP_NAME
	# Create a new user with name USER_NAME
	adduser --disabled-password --uid $USER_ID --gid $GROUP_ID --shell /bin/bash $USER_NAME

	# Set password for a user
	# RUN echo "$USER_NAME:PASSWORD" | chpasswd 

	# Add a user as a new sudoer
	adduser $USER_NAME sudo

	# Remove password
	echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers;
fi