FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV HTTP_PROXY=http://proxy.ign.fr:3128
ENV HTTPS_PROXY=http://proxy.ign.fr:3128
ENV NO_PROXY=localhost,127.0.0.1,*.ign.fr

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3-tk    
# Python deps (minimal copy ONLY for caching)
COPY requirements.txt /tmp/requirements.txt
COPY dust3r/requirements.txt /tmp/dust3r_requirements.txt
COPY dust3r/requirements_optional.txt /tmp/dust3r_requirements_optional.txt

RUN pip install -r /tmp/dust3r_requirements.txt
RUN pip install -r /tmp/dust3r_requirements_optional.txt 
RUN pip install -r /tmp/requirements.txt
RUN pip install opencv-python==4.8.0.74
RUN pip install "gradio>=4.0,<6.0"

RUN apt-get update && apt-get install -y \
    ninja-build \
    cmake \
    libopenblas-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

#RUN which nvcc && nvcc --version
# RUN pip install pytorch3d
RUN git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d && \
    cd /opt/pytorch3d && \
    git checkout v0.7.9 && \
    export USE_CUDA=1 && \
    export FORCE_CUDA=1 && \
    export CUDA_HOME=/usr/local/cuda && \
    export TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0" && \
    pip install -e .

RUN pip install open3d    

# Runtime location (mounted repo)
WORKDIR /mast3r_ign
