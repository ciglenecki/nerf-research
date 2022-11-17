# we want to install as little as possible so that people without docker can use Python via requirements.txt
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Update and upgrade all packages
RUN apt update -y
RUN apt upgrade -y

# Install python
RUN apt install -y git python3 software-properties-common python3-pip python-is-python3

# Install other useful
RUN apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    htop \
    ffmpeg \
    wget

# Colmap dependencies
RUN apt-get install -y \
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
    libcgal-dev

# ======= Ceras Solver ======= http://ceres-solver.org/installation.html
RUN cd /root && wget http://ceres-solver.org/ceres-solver-2.1.0.tar.gz

# Install Ceras dependencies
RUN apt-get install -y cmake libgoogle-glog-dev libgflags-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev

# Install Ceras 
RUN cd /root && tar zxf ceres-solver-2.1.0.tar.gz && \
    mkdir -p ceres-bin &&\
    cd ceres-bin &&\
    cmake ../ceres-solver-2.1.0 &&\
    make -j$(nproc) &&\
    make install

# Download Colmap
RUN cd /root && git clone https://github.com/colmap/colmap.git && \
    cd colmap &&\
    git checkout dev &&\
    mkdir build &&\
    cd build &&\
    cmake .. &&\
    make -j &&\
    make install

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt && pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Set the pretty and obvious prompt
ENV TERM xterm-256color

RUN echo 'export PS1="🐳  \[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]"' >> /root/.bashrc

CMD ["bash", "-l"]