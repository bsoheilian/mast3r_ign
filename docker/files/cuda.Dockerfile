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

# Runtime location (mounted repo)
WORKDIR /mast3r_ign
