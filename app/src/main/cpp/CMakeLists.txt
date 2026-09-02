cmake_minimum_required(VERSION 3.22.1)

project(localmind VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)

# ==========================================================================
# 目标 ABI
#
# 本应用为第五代骁龙 8 至尊版（SM8850 / arm64-v8a）专门调优。
# x86_64 仅为模拟器调试保留，不启用 KleidiAI 等 ARM 专属优化。
# ==========================================================================
if(DEFINED ANDROID_ABI)
    message(STATUS "LocalMind target ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI ON)
        set(GGML_OPENMP ON)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(FATAL_ERROR "LocalMind 只支持 arm64-v8a 与 x86_64，当前: ${ANDROID_ABI}")
    endif()
endif()

# ==========================================================================
# llama.cpp 构建开关
#
# LLAMA_BUILD_MTMD: 构建 libmtmd —— 多模态（图片/音频）推理依赖它。
#   该开关独立于 LLAMA_BUILD_TOOLS，可在不编译 server / cli 等可执行文件
#   的前提下只拿到多模态库。
# ==========================================================================
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp)

if(NOT EXISTS ${LLAMA_SRC}/CMakeLists.txt)
    message(FATAL_ERROR
            "未找到 llama.cpp 源码: ${LLAMA_SRC}\n"
            "请先执行脚本拉取依赖: ./scripts/fetch-llama-cpp.sh")
endif()

set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_MTMD      ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_POC       OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL            OFF CACHE BOOL "" FORCE)

# ==========================================================================
# OpenCL 后端（Adreno GPU）
#
# GGML_OPENCL_USE_ADRENO_KERNELS: 启用针对 Adreno 的 matmul 优化内核。
# GGML_OPENCL_EMBED_KERNELS:      把 .cl 内核源码编译进 so。
#   Android 上必须开启 —— 设备文件系统里不存在内核源文件，
#   运行时无法像桌面那样从磁盘加载。
# ==========================================================================
set(GGML_OPENCL                     ON  CACHE BOOL "" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS  ON  CACHE BOOL "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS       ON  CACHE BOOL "" FORCE)
set(GGML_OPENCL_TARGET_VERSION      300 CACHE STRING "" FORCE)

# QNN（Hexagon NPU）后端默认关闭。
# 需要 Qualcomm QNN SDK 且需单独接受许可，开启方式见 docs/SNAPDRAGON.md。
set(GGML_QNN OFF CACHE BOOL "" FORCE)


# ==========================================================================
# OpenCL 头文件与库（Android / CI 适配）
#
# llama.cpp 的 ggml-opencl 后端执行 find_package(OpenCL REQUIRED)，
# 而 FindOpenCL 默认只搜桌面标准路径，在 Android NDK 环境下找不到：
#   - 头文件：CI 工作流已预下载 Khronos OpenCL-Headers（OPENCL_HEADERS_DIR），
#     否则回退到 NDK sysroot 自带的 CL/ 头（若存在）。
#   - 库：链接 NDK platforms 下的 libOpenCL.so stub；
#     运行时由设备系统（Adreno）加载真正的 /system/lib64/libOpenCL.so。
# ==========================================================================
if(DEFINED ENV{OPENCL_HEADERS_DIR})
    if(EXISTS "$ENV{OPENCL_HEADERS_DIR}/CL/cl.h")
        set(OpenCL_INCLUDE_DIR "$ENV{OPENCL_HEADERS_DIR}")
    endif()
elseif(DEFINED ANDROID_NDK AND EXISTS "${ANDROID_NDK}/sysroot/usr/include/CL/cl.h")
    set(OpenCL_INCLUDE_DIR "${ANDROID_NDK}/sysroot/usr/include")
endif()
if(DEFINED ANDROID_NDK)
    file(GLOB _lm_ocl_libs
        "${ANDROID_NDK}/platforms/android-*/arch-${ANDROID_ABI}/usr/lib/libOpenCL.so")
    if(_lm_ocl_libs)
        list(GET _lm_ocl_libs 0 OpenCL_LIBRARY)
    endif()
    unset(_lm_ocl_libs)
endif()
set(OpenCL_INCLUDE_DIR "${OpenCL_INCLUDE_DIR}" CACHE PATH "OpenCL headers dir" FORCE)
set(OpenCL_LIBRARY     "${OpenCL_LIBRARY}"     CACHE FILEPATH "OpenCL stub lib" FORCE)
message(STATUS "LocalMind OpenCL_INCLUDE_DIR=${OpenCL_INCLUDE_DIR}")
message(STATUS "LocalMind OpenCL_LIBRARY=${OpenCL_LIBRARY}")

add_subdirectory(${LLAMA_SRC} build-llama)

# ==========================================================================
# LocalMind 原生库
# ==========================================================================
add_library(localmind SHARED
        lm_engine.cpp
        lm_jni.cpp)

target_compile_definitions(localmind PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_CPU_KLEIDIAI=$<BOOL:${GGML_CPU_KLEIDIAI}>
        GGML_OPENMP=$<BOOL:${GGML_OPENMP}>)

target_include_directories(localmind PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/tools/mtmd
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
        # JNI 层用 nlohmann::json 传递配置与统计（common 库自身也依赖它）
        ${LLAMA_SRC}/vendor)

target_link_libraries(localmind
        llama
        llama-common
        mtmd
        android
        log)
