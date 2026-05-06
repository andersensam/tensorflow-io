licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "az",
    srcs = [
        "az_filesystem.cc",
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
        "@com_github_azure_azure_sdk_for_cpp//:azure",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@xla//xla/tsl/platform:status",
        "@xla//xla/tsl/platform:logging",
        "@org_tensorflow//tensorflow/c:tf_file_statistics",
        "@org_tensorflow//tensorflow/c/experimental/filesystem:modular_filesystem",
    ],
    alwayslink = 1,
)
