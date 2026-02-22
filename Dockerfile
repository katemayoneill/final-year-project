# dockerfile for openpose work

# nvidia cuda base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# give this a go if stuff isnt working properly with interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# installing dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-dev \
    libatlas-base-dev \
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libboost-all-dev \
    python3-dev \
    python3-pip \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /openpose

# clone openpose
RUN git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git . && \
    git submodule update --init --recursive

# copy pretrained models into the proper directories
RUN mkdir -p models/pose/body_25 && \
    mkdir -p models/pose/coco && \
    mkdir -p models/pose/mpi && \
    mkdir -p models/face && \
    mkdir -p models/hand

COPY models/ /openpose/models/

RUN cd models/pose/body_25 && \
    wget -q https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/body_25/pose_deploy.prototxt && \
    cd ../coco && \
    wget -q https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt && \
    cd ../mpi && \
    wget -q https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/mpi/pose_deploy_linevec.prototxt && \
    cd ../../face && \
    wget -q https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/face/pose_deploy.prototxt && \
    cd ../hand && \
    wget -q https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt

# create build directory and compile openpose
RUN mkdir build && cd build && \
    cmake \
    -DBUILD_PYTHON=ON \
    -DUSE_CUDNN=ON \
    -DGPU_MODE=CUDA \
    -DCUDA_ARCH=Manual \
    -DCUDA_ARCH_BIN="60 61 62 70 72 75 80 86 89 90" \
    -DCUDA_ARCH_PTX="90" \
    .. && \
    make -j$(nproc)

# install python dependencies
RUN pip3 install numpy opencv-python

ENV PYTHONPATH="/openpose/build/python"
ENV LD_LIBRARY_PATH="/openpose/build/src/openpose:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"

WORKDIR /app

COPY process.py /app/process.py
COPY annotate.py /app/annotate.py

CMD ["sleep", "infinity"]