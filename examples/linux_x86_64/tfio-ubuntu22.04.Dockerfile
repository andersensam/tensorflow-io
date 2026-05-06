# syntax=docker/dockerfile:1

ARG TARGET=base
ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} AS python
# Build Python 3.12
RUN mkdir -p /tmp/staging
WORKDIR /tmp/staging
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
        libnss3-dev libssl-dev libreadline-dev libffi-dev pkg-config wget \
        libbz2-dev liblzma-dev libsqlite3-dev uuid-dev libgdbm-compat-dev \
        tk-dev libnsl-dev curl gnupg && \
    apt clean -y && \
    curl -o Python-3.12.11.tgz https://www.python.org/ftp/python/3.12.11/Python-3.12.11.tgz && \
    tar -xvf Python-3.12.11.tgz && \
    ./Python-3.12.11/configure --enable-optimizations --with-ensurepip=install --prefix=/opt/python3.12 && \
    make all -j22 && \
    make altinstall -j22 && \
    apt-get remove -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
        libnss3-dev libssl-dev libreadline-dev libffi-dev pkg-config wget \
        libbz2-dev liblzma-dev libsqlite3-dev uuid-dev libgdbm-compat-dev \
        tk-dev libnsl-dev curl gnupg && \
    apt-get autoremove -y && \
    apt clean -y && \
    rm -rf ./*

# Start with a clean Ubuntu 22.04 image and copy the Python 3.12 installation from the previous builder image
FROM ${BASE_IMAGE} AS base
RUN mkdir -p /tmp/staging && mkdir -p /opt/python3.12
WORKDIR /tmp/staging
# Add the Python 3.12 install to this builder stage
COPY --from=python /opt/python3.12 /opt/python3.12
# Extract LLVM
ADD LLVM-20.1.7-Linux-X64.tar.xz /tmp/staging/

# Setup the virtual environment for building
ENV VIRTUAL_ENV=/opt/venv
RUN /opt/python3.12/bin/python3.12 -m venv ${VIRTUAL_ENV}
ENV PATH="$VIRTUAL_ENV/bin:/tmp/staging/LLVM-20.1.7-Linux-X64/bin:$PATH"
ENV LLVM_HOME=/tmp/staging/LLVM-20.1.7-Linux-X64 CUDA_HOME=/usr/local/cuda-12.8

# Enable the CUDA repository and install the required libraries for building TensorFlow
RUN apt-get update && apt-get install -y curl && \
    curl -o cuda-keyring_1.1-1_all.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y cuda-libraries-dev-12-8 libcudnn9-dev-cuda-12 libnccl-dev ibverbs-utils \
         patchelf wget curl llvm build-essential git \ 
         cuda-nvvm-12-8 cuda-nvml-dev-12-8 cuda-nvrtc-dev-12-8 cuda-nvcc-12-8 libnccl2 \
         cuda-cupti-12-8 cuda-cupti-dev-12-8 && \
    apt clean -y

# Prepare to build and set any environmental flags that bazel might be difficult with
ENV CC_OPT_FLAGS="-Wno-gnu-offsetof-extensions -Wno-error -Wno-c23-extensions -Wno-macro-redefined" CPATH="${CUDA_HOME}/include:/usr/local/cuda-12.8/targets/x86_64-linux/include"

# Install Bazelisk (Bazel wrapper), using a local bazel file since the download doesn't work half the time
COPY bazel /usr/local/bin/bazel
RUN chmod +x /usr/local/bin/bazel && /usr/local/bin/bazel version && mkdir -p /workspace

WORKDIR /workspace
# Fetch the correct branch, grab TensorFlow, install our custom version
# then create the tirpc symlink needed for compilation
RUN git clone --depth 1 --branch tensorflow_r2.21.0.1 https://github.com/andersensam/tensorflow-io && \
    pip install --upgrade pip uv && pip cache purge && \
    uv pip install tensorflow==2.21.0 setuptools wheel && \
    uv pip uninstall tensorflow && \
    uv pip install --no-deps --no-index --find-links https://storage.googleapis.com/axlearn-wheels/wheels.html tensorflow==2.21.0.1 && \
    uv cache clean && \
    ln -s /usr/include/tirpc /workspace/tensorflow-io/third_party/tirpc

WORKDIR /workspace/tensorflow-io
COPY tfio_py3.12.brc .bazelrc
# Run the build with all flags we found necessary
RUN bazel build --noenable_bzlmod --copt="-fPIC" --verbose_failures --spawn_strategy=local \
    --copt=-Ithird_party/tirpc --linkopt=-fuse-ld=gold \
    --per_file_copt=third_party/.*,external/.*@-Wno-error \
    --per_file_copt=third_party/.*,external/.*@-Wno-implicit-function-declaration \
    --per_file_copt=third_party/.*,external/.*@-Wno-int-conversion \
    --per_file_copt=third_party/.*,external/.*@-Wno-enum-constexpr-conversion \
    --per_file_copt=third_party/.*,external/.*@-Wno-private-header \
    -- "//tensorflow_io:python/ops/libtensorflow_io.so" "//tensorflow_io:python/ops/libtensorflow_io_plugins.so" \
    "//tensorflow_io_gcs_filesystem/..."

# Build the wheels
RUN rm -rf build && \
    mkdir build && \
    cp -r -L bazel-bin/tensorflow_io build/tensorflow_io && \
    python3 setup.py --data build bdist_wheel && \
    rm -rf build && \
    mkdir build && \
    cp -r -L bazel-bin/tensorflow_io_gcs_filesystem  build/tensorflow_io_gcs_filesystem && \
    python3 setup.py --data build bdist_wheel --project tensorflow-io-gcs-filesystem && \
    mkdir -p /mnt/export && cp dist/*.whl /mnt/export

FROM scratch AS target
COPY --from=base /workspace/tensorflow-io/dist /wheels