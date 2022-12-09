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
    wget \
    gcc \
    g++

# Install colmap
RUN apt-get install -y colmap

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt && pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Set the pretty and obvious prompt
ENV TERM xterm-256color

RUN echo 'export PS1="🐳  \[\033[1;36m\]\h \[\033[1;34m\]\W\[\033[0;35m\] \[\033[1;36m\]# \[\033[0m\]"' >> /root/.bashrc

CMD ["bash", "-l"]