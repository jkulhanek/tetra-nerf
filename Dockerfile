# NOTE: The OptiX library has to be downloaded separatedly due to licensing.
#       Please download the OptiX library (>=7.2, <=7.6) from https://developer.nvidia.com/designworks/optix/downloads/legacy
#       Due to the OptiX library residing outside the build context, Dockerfile requires docker buildx
#       to build the image. Please refer to https://docs.docker.com/buildx/working-with-buildx/ 
#       for more information.  
# The following command builds the image:
#     `docker build -t tetra-nerf:latest --build-context optix=/opt/optix
# The image can be run with the following command:
#     `docker run -it -e  NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility --gpus all tetra-nerf:latest`
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
COPY --from=optix . /opt/optix
RUN if [ ! -e /opt/optix/include/optix.h ]; then echo "Could not find the OptiX library. Please install the Optix SDK and add the following argument to the buildx command: --build-context optix=/path/to/the/SDK"; exit 1; fi && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    cmake \
    git \
    colmap \
    ffmpeg \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    libcgal-dev \
    && \
    rm -rf /var/lib/apt/lists/*

RUN export PIP_ROOT_USER_ACTION=ignore && \
    pip install --upgrade pip && \
    pip uninstall -y functorch && \
    pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install nerfstudio==0.2.0 trimesh==3.21.5 dm-pix==0.4.0  && \
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip cache purge

ARG CUDAARCHS=61;70;75;80;86
ENV CUDAARCHS=${CUDAARCHS} \
    TCNN_CUDA_ARCHITECTURES=${CUDAARCHS} \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility
# Basically just to remove the NerfStudio warning
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

LABEL org.opencontainers.image.authors="jonas.kulhanek@live.com"
ENV PATH="/home/user/.local/bin:${PATH}"
RUN adduser --disabled-password user --gecos "First Last,RoomNumber,WorkPhone,HomePhone"
USER user

WORKDIR /home/user
COPY --chown=user . /home/user/tetra-nerf
RUN pip install -e tetra-nerf
