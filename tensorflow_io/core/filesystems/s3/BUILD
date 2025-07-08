licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "s3",
    srcs = [
        "aws_logging.cc",
        "aws_logging.h",
        "s3_filesystem.cc",
        "s3_filesystem.h",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@aws-sdk-cpp//:s3",
        "@aws-sdk-cpp//:transfer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@org_tensorflow//third_party/xla/third_party/tsl/tsl/platform:status",
        "@org_tensorflow//tensorflow/c:logging",
        "@org_tensorflow//tensorflow/c:c_api_macros",
        "@org_tensorflow//tensorflow/c:tf_file_statistics",
        "@org_tensorflow//tensorflow/c/experimental/filesystem:modular_filesystem",
    ],
    alwayslink = 1,
)
