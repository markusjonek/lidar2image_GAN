FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /lidar_img

# Update the package list
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    git \
    libpcl-dev \
    libopencv-dev \
    wget \
    unzip \
    cmake \
    && apt-get clean


# Install python packages
RUN pip3 install --upgrade pip
RUN pip3 install numpy \
    matplotlib \
    opencv-python \
    torch \
    torch_geometric \
    wandb

