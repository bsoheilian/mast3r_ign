FROM nvcr.io/nvidia/pytorch:25.03-py3

LABEL description="Docker container for MASt3R with dependencies installed. CUDA VERSION"
ENV DEVICE="cuda"
ENV MODEL="MASt3R_ViTLarge_BaseDecoder_512_dpt.pth"
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --recursive https://github.com/naver/mast3r /mast3r
WORKDIR /mast3r/dust3r
RUN pip install -r requirements.txt
RUN pip install -r requirements_optional.txt
RUN pip install opencv-python==4.8.0.74

WORKDIR /mast3r/dust3r/croco/models/curope/
RUN python setup.py build_ext --inplace

WORKDIR /mast3r
RUN pip install -r requirements.txt
RUN pip install "gradio>=4.0,<6.0"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
