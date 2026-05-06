# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=ubuntu:20.04

# Start with a clean Ubuntu 20.04 image and copy the Python 3.12 installation from the previous builder image
# and then add LLVM
FROM ${BASE_IMAGE} AS base
RUN mkdir -p /tmp/staging && mkdir -p /opt/python3.12
WORKDIR /tmp/staging
# Add the Python 3.12 install to this builder stage (build first with python-3.12-ubuntu20.04.Dockerfile)
COPY --from=python:3.12-ubuntu20.04 /python3.12 /opt/python3.12
# Copy the LLVM 20.1.7 install (build first with llvm-20.1.7-ubuntu20.04.Dockerfile)
COPY --from=llvm:20.1.7-ubuntu20.04 /llvm /opt/llvm
# Setup the virtual environment for building
ENV VIRTUAL_ENV=/opt/venv
RUN /opt/python3.12/bin/python3.12 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:/opt/llvm/bin:$PATH"
ENV LLVM_HOME=/opt/llvm CUDA_HOME=/usr/local/cuda-12.8

# Enable the CUDA repository and install the required libraries for building TensorFlow
RUN apt-get update && apt-get install -y curl && \
    curl -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-libraries-dev-12-8 libcudnn9-dev-cuda-12 libnccl-dev ibverbs-utils \
         patchelf wget curl llvm build-essential git \ 
         cuda-nvvm-12-8 cuda-nvml-dev-12-8 cuda-nvrtc-dev-12-8 cuda-nvcc-12-8 libnccl2 \
         cuda-cupti-12-8 cuda-cupti-dev-12-8 libxml2-dev libssl-dev && \
    apt clean -y

# Prepare to build and set any environmental flags that bazel might be difficult with
ENV CC_OPT_FLAGS="-Wno-gnu-offsetof-extensions -Wno-error -Wno-c23-extensions -Wno-macro-redefined" CPATH="${CUDA_HOME}/include:/usr/local/cuda-12.8/targets/x86_64-linux/include"

# Install Bazelisk (Bazel wrapper), using a local bazel file since the download doesn't work half the time
COPY bazel /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel && /usr/local/bin/bazel version && mkdir -p /workspace

WORKDIR /workspace
RUN git clone --depth 1 https://github.com/andersensam/tensorflow-io && \
    pip install --upgrade pip && pip install uv && pip cache purge && \
    uv pip install tensorflow==2.19.1 setuptools && \
    uv pip uninstall tensorflow && \
    uv pip install --no-deps --no-index https://storage.googleapis.com/axlearn-wheels/tensorflow/tensorflow-2.19.1.3-cp312-cp312-manylinux_2_31_x86_64.whl && \
    uv cache clean

WORKDIR /workspace/tensorflow-io
COPY tfio_py3.12.brc .bazelrc
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