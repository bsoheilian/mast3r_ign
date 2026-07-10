FROM nvcr.io/nvidia/pytorch:25.03-py3

ARG DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    bash-completion \
    vim \
    less \
    nano \    
    git \
    libglib2.0-0 \
    build-essential \
    ninja-build \
    cmake \
    libopenblas-dev \
    libgl1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN touch /root/.bashrc
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN echo "source /etc/bash_completion" >> /root/.bashrc && \
    echo "alias ll='ls -alF --color=auto'" >> /root/.bashrc && \
    echo "alias la='ls -A --color=auto'" >> /root/.bashrc && \
    echo "alias grep='grep --color=auto'" >> /root/.bashrc

# Python deps (minimal copy ONLY for caching)
COPY requirements.txt /tmp/requirements.txt
COPY dust3r/requirements.txt /tmp/dust3r_requirements.txt
COPY dust3r/requirements_optional.txt /tmp/dust3r_requirements_optional.txt

RUN pip install \
    -r /tmp/dust3r_requirements.txt \
    -r /tmp/dust3r_requirements_optional.txt \
    -r /tmp/requirements.txt \
    opencv-python==4.8.0.74 \
    "gradio>=4.0,<6.0" \
    open3d




RUN git clone https://github.com/facebookresearch/pytorch3d.git /opt/pytorch3d && \
    cd /opt/pytorch3d && \
    git checkout v0.7.9 && \
    export USE_CUDA=1 && \
    export FORCE_CUDA=1 && \
    export CUDA_HOME=/usr/local/cuda && \
    export TORCH_CUDA_ARCH_LIST="12.0" && \
    pip install -e .

  

# Runtime location (mounted repo)
WORKDIR /mast3r_ign
