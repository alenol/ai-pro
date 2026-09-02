#!/usr/bin/env bash
# 拉取 llama.cpp 源码到 third_party/llama.cpp（CMake 通过相对路径引用它）。
# 国内网络若直连缓慢，可设置环境变量 LLAMACPP_MIRROR 指向镜像后重跑。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/third_party/llama.cpp"

if [ -f "$DEST/CMakeLists.txt" ]; then
    echo "llama.cpp 已存在: $DEST (跳过)"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"

UPSTREAM="https://github.com/ggml-org/llama.cpp.git"
URL="${LLAMACPP_MIRROR:-$UPSTREAM}"

echo "克隆 llama.cpp -> $DEST"
echo "源: $URL"
git clone --depth 1 "$URL" "$DEST" || {
    echo "直连失败。可尝试镜像，例如：" >&2
    echo "  LLAMACPP_MIRROR=https://ghfast.top/https://github.com/ggml-org/llama.cpp.git ./scripts/fetch-llama-cpp.sh" >&2
    exit 1
}

echo "完成: $DEST"
