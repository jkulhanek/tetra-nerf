# NOTE: The OptiX library has to be downloaded separatedly due to licensing.
#       Please download the OptiX library (>=7.2, <=7.6) from https://developer.nvidia.com/designworks/optix/downloads/legacy
#       Due to the OptiX library residing outside the build context, Dockerfile requires docker buildx
#       to build the image. Please refer to https://docs.docker.com/buildx/working-with-buildx/ 
#       for more information.  
# The following command builds the image:
#     `docker build -t tetra-nerf:latest --build-context optix=/opt/optix
# The image can be run with the following command:
#     `docker run -it --gpus all tetra-nerf:latest`
FROM dromni/nerfstudio:0.3.4
COPY --from=optix . /opt/optix
RUN if [ ! -e /opt/optix/include/optix.h ]; then echo "Could not find the OptiX library. Please install the Optix SDK and add the following argument to the buildx command: --build-context optix=/path/to/the/SDK"; exit 1; fi

ARG CUDAARCHS=61;70;75;80;86
ENV CUDAARCHS=${CUDAARCHS} \
    TCNN_CUDA_ARCHITECTURES=${CUDAARCHS} \
    NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
    CMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN export PIP_ROOT_USER_ACTION=ignore && \
    python3.10 -m pip install --upgrade "jax[cpu]" dm_pix && \
    python3.10 -m pip cache purge

LABEL org.opencontainers.image.authors="jonas.kulhanek@live.com"

COPY --chown=1000 . /home/user/tetra-nerf

USER 1000
RUN path="$PWD" && cd /home/user/tetra-nerf && \
    python3.10 -m pip install -e . && \
    python3.10 -m pip install pytest && \
    python3.10 -m pip cache purge && \
    cd "$path"

# Install completions, lerf loads GPU so we just remove it for install the completions
RUN pip uninstall -y lerf && \
    ns-install-cli --mode install

# Remove /opt/optix due to its license
# Also link python as python3 (could be done with python-is-python3)
USER root
RUN rm -rf /opt/optix && \
    [ -e /usr/bin/python ] || ln -s /usr/bin/python3 /usr/bin/python
RUN chmod -R go=u /home/user
USER 1000

CMD /bin/bash -l
