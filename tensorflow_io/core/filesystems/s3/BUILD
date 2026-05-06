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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/log:log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/container:btree",
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@aws-sdk-cpp//:s3",
        "@aws-sdk-cpp//:transfer",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:logging",
        "@org_tensorflow//tensorflow/c:c_api_macros",
        "@org_tensorflow//tensorflow/c:tf_file_statistics",
        "@org_tensorflow//tensorflow/c/experimental/filesystem:modular_filesystem",
    ],
    alwayslink = 1,
)
