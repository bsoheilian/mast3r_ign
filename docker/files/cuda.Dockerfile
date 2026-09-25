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
    libegl1 \
    libusb-1.0-0 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.bash_history_data && touch /root/.bashrc /root/.bash_history_data/history
RUN sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN echo 'if [ -f /etc/bash_completion ]; then' >> /root/.bashrc && \
    echo '    source /etc/bash_completion' >> /root/.bashrc && \
    echo 'fi' >> /root/.bashrc && \
    echo 'set -o emacs' >> /root/.bashrc && \
    echo 'export HISTFILE=/root/.bash_history_data/history' >> /root/.bashrc && \
    echo 'export HISTSIZE=10000' >> /root/.bashrc && \
    echo 'export HISTFILESIZE=20000' >> /root/.bashrc && \
    echo 'shopt -s histappend' >> /root/.bashrc && \
    echo 'PROMPT_COMMAND="history -a; history -n${PROMPT_COMMAND:+; $PROMPT_COMMAND}"' >> /root/.bashrc && \
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
    open3d \
    rasterio

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
