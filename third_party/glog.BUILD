licenses(["notice"])

exports_files(["COPYING"])

load(":bazel/glog.bzl", "glog_library")

glog_library()

# platform() to build with clang-cl on Bazel CI. This is enabled with
# the flags in .bazelci/presubmit.yml:
#
#   --incompatible_enable_cc_toolchain_resolution
#   --extra_toolchains=@local_config_cc//:cc-toolchain-x64_windows-clang-cl
#   --extra_execution_platforms=//:x64_windows-clang-cl
platform(
    name = "x64_windows-clang-cl",
    constraint_values = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        "@rules_cc//cc/private/toolchain:clang-cl",
    ],
)