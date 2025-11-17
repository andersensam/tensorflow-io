# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} AS base
RUN mkdir -p /tmp/staging
WORKDIR /tmp/staging
# Install python3.10
RUN apt-get update && apt-get upgrade -y && apt-get install -y python3.10-venv python3.10-dev \
    && apt clean -y
# Extract LLVM
ADD LLVM-20.1.7-Linux-X64.tar.xz /tmp/staging/

# Setup the virtual environment for building
ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:/tmp/staging/LLVM-20.1.7-Linux-X64/bin:$PATH"
ENV LLVM_HOME=/tmp/staging/LLVM-20.1.7-Linux-X64 CUDA_HOME=/usr/local/cuda-12.8

# Enable the CUDA repository and install the required libraries (libnvrtc.so)
RUN apt-get update && apt-get install -y curl && \
    curl -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y cuda-libraries-dev-12-8 libcudnn9-dev-cuda-12 libnccl-dev ibverbs-utils \
         patchelf wget curl llvm build-essential git \ 
         cuda-nvvm-12-8 cuda-nvml-dev-12-8 cuda-nvrtc-dev-12-8 cuda-nvcc-12-8 libnccl2 \
         cuda-cupti-12-8 cuda-cupti-dev-12-8 && \
    apt clean -y

# Prepare to build
ENV CC_OPT_FLAGS="-Wno-gnu-offsetof-extensions -Wno-error -Wno-c23-extensions -Wno-macro-redefined" CPATH="${CUDA_HOME}/include:/usr/local/cuda-12.8/targets/x86_64-linux/include"

# Install Bazelisk (Bazel wrapper), using a local bazel file since the download doesn't work half the time
COPY bazel /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel && /usr/local/bin/bazel version && mkdir -p /workspace

WORKDIR /workspace
RUN git clone --depth 1 https://github.com/andersensam/tensorflow-io && \
    pip install --upgrade pip && pip install uv && pip cache purge && \
    uv pip install tensorflow==2.19.1 setuptools && \
    uv pip uninstall tensorflow && \
    uv pip install --no-deps --no-index --find-links https://storage.googleapis.com/axlearn-wheels/wheels.html tensorflow==2.19.1.2 && \
    uv cache clean

WORKDIR /workspace/tensorflow-io
COPY tfio_py3.10.brc .bazelrc
RUN bazel build --copt="-fPIC"  --verbose_failures --spawn_strategy=local \
    --copt=-I/usr/include/tirpc --linkopt=-fuse-ld=gold \
    --per_file_copt=third_party/.*,external/.*@-Wno-error \
    -- "//tensorflow_io:python/ops/libtensorflow_io.so" "//tensorflow_io:python/ops/libtensorflow_io_plugins.so" \
    "//tensorflow_io_gcs_filesystem/..."

RUN python3 setup.py --data bazel-bin bdist_wheel && \
    python3 setup.py --data bazel-bin bdist_wheel --project tensorflow-io-gcs-filesystem && \
    mkdir -p /mnt/export && cp dist/*.whl /mnt/export

FROM scratch AS target
COPY --from=base /workspace/tensorflow-io/dist /wheels