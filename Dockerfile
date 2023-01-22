# Define base image.
FROM nvidia/cuda:11.7.0-devel-ubuntu22.10
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Zagreb
## CUDA architectures, required by tiny-cuda-nn.
ENV TCNN_CUDA_ARCHITECTURES=61
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"

# Update and upgrade all packages
RUN apt update -y
RUN apt upgrade -y

# Install python
RUN apt install -y git python3 software-properties-common python3-pip python-is-python3

# Install deps
RUN apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    htop \
    ffmpeg \
    wget \
    gcc \
    g++ \
    git \
    cmake \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libqt5gui5 \
    qt5-qmake \
    libxcb-util-dev \
    libprotobuf-dev \
    libatlas-base-dev

# Install GLOG (required by ceres).
RUN git clone --branch v0.6.0 https://github.com/google/glog.git --single-branch && \
    cd glog && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install && \
    cd ../.. && \
    rm -r glog

# Add glog path to LD_LIBRARY_PATH.
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

# # Ceres solver
RUN apt-get install libatlas-base-dev libsuitesparse-dev &&\
    wget https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.1.0.tar.gz &&\
    tar zxf 2.1.0.tar.gz &&\
    cd ceres-solver-2.1.0 &&\
    mkdir build &&\
    cd build &&\
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF &&\
    make -j 2 &&\
    make install

# Compile Colamp
RUN git clone https://github.com/colmap/colmap.git &&\
    cd colmap &&\
    git checkout dev &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j 2 &&\
    make install

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install python packets
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Dependency for nerfstudio that has to be installed AFTER torch
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Arguments required for setting the permissions
ARG USER_NAME=root
ARG USER_ID
ARG GROUP_NAME
ARG GROUP_ID

# Create account in Docker Container with the same name and group as the current user who is building the Docker image.
COPY docker_set_perms.sh /tmp/docker_set_perms.sh
RUN chmod +x /tmp/docker_set_perms.sh && /tmp/docker_set_perms.sh $USER_ID $USER_NAME $GROUP_ID $GROUP_NAME

USER $USER_NAME

# Set the pretty and obvious prompts
ENV TERM xterm-256color
RUN echo 'export PS1="\A \[\033[1;36m\]\h\[\033[1;34m\] \w \[\033[0;015m\]\\$ \[$(tput sgr0)\]\[\033[0m\]"' >> ~/.bashrc

# Set bash entrypoint location to home directory
# WORKDIR $USER_HOME

CMD ["bash", "-l"]