# Compiling and Installing TensorFlow-IO

Much of the build process is the same as TensorFlow itself [documented here](https://github.com/andersensam/tensorflow/blob/r2.19/examples/README.md).

For simplicity, the work root setup will not be repeated in this README.

## Building for `linux_x86_64` or `linux_arm64`

Ensure your container runtime of choice is installed and properly configured.

Since my target OS is Ubuntu 22.04 and Python 3.12 is not available by default, I build it first in a separate image and copy its contents to the TensorFlow-IO builder image. Once Python 3.12 is built, it is installed to `/opt/python3.12` and a virtual environment is created in `/opt/venv`. Please modify the Python compilation to account for the number of CPUs on your build machine.


With all the files in place, we are ready for compilation. In the work root:
```
podman build . -f tfio-ubuntu22.04.Dockerfile -t tensorflow-io:ubuntu22.04
```

The final instructions of the Dockerfile copy the wheels to blank image:
```
FROM scratch AS target
COPY --from=base /mnt/export /wheels
```

The above can be removed if desired, ensuring the build context is fully saved and wheels are accessible at `/mnt/export`.

Assuming the default config, with the image build complete, the image `tensorflow-io:ubuntu22.04` has its wheels stored in `/wheels`.

### Note on Python 3.10

Follow the above process, using `tfio-ubuntu22.04-python3.10.Dockerfile` to build instead.

## Building for `macos_arm64`

Unlike the Linux targets, the `macos_arm64` target does not use containers.

Ensure Xcode is installed and configured properly. LLVM must also be downloaded and extracted. Install the desired Python version and set up a virtual environment. Also ensure `git` is installed.

My setup might look something like:
```
python3.12 -m venv .venv
source .venv/bin/activate
```

Ensure the right version of TensorFlow is installed
```
pip install --upgrade pip 
pip install uv
uv pip install tensorflow==2.19.1
```

Run the configure script
```
./configure
```

Edit `.bazelrc` to ensure the right macos targets are present:
```
build:macos --copt="-DGRPC_BAZEL_BUILD"
build:macos --copt="-D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION"
build:macos --action_env MACOSX_DEPLOYMENT_TARGET=12.0
build:macos --define=grpc_no_ares=true
build:macos --copt=-Wno-traditional
```

Examples of `.bazelrc` are availabe in the `macos_arm64` directory, titled `tfio_py3.12.brc` and `tfio_py3.10.brc`

Ensure the following lines are properly edited:
```
build --action_env TF_HEADER_DIR="PATH TO venv/lib/python3.12/site-packages/tensorflow/include"
build --action_env TF_SHARED_LIBRARY_DIR="PATH TO venv/lib/python3.12/site-packages/tensorflow"
```

Edited to:
```
build --action_env TF_HEADER_DIR="/Users/sam/.venv/lib/python3.12/site-packages/tensorflow/include"
build --action_env TF_SHARED_LIBRARY_DIR="/Users/sam/.venv/lib/python3.12/site-packages/tensorflow"
```

Build and package
```
bazel build --copt="-fPIC"  --verbose_failures --spawn_strategy=local \
    --per_file_copt="third_party/.*,external/.*@-Wno-error" \
    --config=macos \
    --experimental_repo_remote_exec \
    -- "//tensorflow_io:python/ops/libtensorflow_io.so" "//tensorflow_io:python/ops/libtensorflow_io_plugins.so" \
    "//tensorflow_io_gcs_filesystem/..."

python3 setup.py --data bazel-bin bdist_wheel
python3 setup.py --data bazel-bin bdist_wheel --project tensorflow-io-gcs-filesystem
```