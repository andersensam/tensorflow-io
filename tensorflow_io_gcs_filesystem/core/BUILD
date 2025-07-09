licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "//:tools/build/tensorflow_io.bzl",
    "tf_io_copts",
)

cc_library(
    name = "gs",
    srcs = [
        "cleanup.h",
        "expiring_lru_cache.h",
        "file_system_plugin_gs.cc",
        "file_system_plugin_gs.h",
        "gcs_filesystem.cc",
        "gcs_helper.cc",
        "gcs_helper.h",
        "ram_file_block_cache.cc",
        "ram_file_block_cache.h",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "@local_config_tf//:stub/libtensorflow_framework.lib",
        ],
        "//conditions:default": [
            "@local_config_tf//:stub/libtensorflow_framework.so",
        ],
    }),
    copts = select({
        "@bazel_tools//src/conditions:windows": [
            "/DEIGEN_STRONG_INLINE=inline", 
            "-DTENSORFLOW_MONOLITHIC_BUILD", 
            "/DPLATFORM_WINDOWS", 
            "/DEIGEN_HAS_C99_MATH", 
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL", 
            "/DEIGEN_AVOID_STL_ARRAY", 
            "/Iexternal/gemmlowp", 
            "/wd4018", 
            "/wd4577", 
            "/DNOGDI", 
            "/UTF_COMPILE_LIBRARY", 
            "/DNDEBUG"], 
        "@bazel_tools//src/conditions:darwin": ["-DNDEBUG"],
        "//conditions:default": ["-DNDEBUG", "-pthread"], "-std=c++17"}),
    linkstatic = True,
    deps = [
        "@com_github_googleapis_google_cloud_cpp//:storage_client",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:variant",
        "@local_config_tf//:tf_c_header_lib",
        "@local_config_tf//:tf_tsl_header_lib",
        "@org_tensorflow//third_party/xla/third_party/tsl/tsl/platform:status",
        "@org_tensorflow//tensorflow/c:c_api_macros",
        "@org_tensorflow//tensorflow/c:env",
        "@org_tensorflow//tensorflow/c:c_api",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "python/ops/libtensorflow_io_gcs_filesystem.so",
    copts = tf_io_copts(),
    linkshared = 1,
    deps = [
        "//tensorflow_io_gcs_filesystem/core:gs",
    ],
)
