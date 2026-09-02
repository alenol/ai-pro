#!/usr/bin/env bash
# LocalMind 一键构建脚本。
#
# 前置条件（本机需自行安装，脚本不会代装）：
#   - Android SDK (cmdline-tools)，且已安装 platforms;android-35、build-tools;35.0.0
#   - NDK 27.0.12077973（或编辑 app/build.gradle.kts 里的 ndkVersion）
#   - CMake 3.22.1（或编辑 app/build.gradle.kts 里的 externalNativeBuild.cmake.version）
#   - JDK 17
#
# 用法：
#   ./scripts/build_apk.sh            # 构建 debug APK
#   ./scripts/build_apk.sh release    # 构建 release APK（需自行配置签名）
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/3 拉取 llama.cpp =="
bash ./scripts/fetch-llama-cpp.sh

echo "== 2/3 确保 gradle wrapper =="
if [ ! -f ./gradlew ]; then
    if command -v gradle >/dev/null 2>&1; then
        gradle wrapper --gradle-version 8.9
    else
        echo "未找到 gradlew，也没有 gradle。请用 Android Studio 打开本工程（会自动生成 wrapper）。" >&2
        exit 1
    fi
fi

echo "== 3/3 编译 APK =="
if [ "${1:-debug}" = "release" ]; then
    ./gradlew :app:assembleRelease
    APK="app/build/outputs/apk/release/app-release.apk"
else
    ./gradlew :app:assembleDebug
    APK="app/build/outputs/apk/debug/app-debug.apk"
fi

if [ -f "$APK" ]; then
    echo "构建成功: $(realpath "$APK")"
else
    echo "未找到产物，请检查上面的构建日志。" >&2
    exit 1
fi
