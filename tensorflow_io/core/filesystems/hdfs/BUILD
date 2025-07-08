licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "hdfs",
    srcs = [
        "hadoop_filesystem.cc",
    ],
    copts = tf_io_copts(),
    linkstatic = True,
    deps = [
        "//tensorflow_io/core/filesystems:filesystem_plugins_header",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@hadoop",
        "@org_tensorflow//third_party/xla/third_party/tsl/tsl/platform:status",
        "@org_tensorflow//tensorflow/c:logging",
        "@org_tensorflow//tensorflow/c:c_api_macros",
        "@org_tensorflow//tensorflow/c:tf_file_statistics",
        "@org_tensorflow//tensorflow/c/experimental/filesystem:modular_filesystem",

    ],
    alwayslink = 1,
)
