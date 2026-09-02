# LocalMind · 安卓本地 AI

一个**完全在设备端运行**的安卓 AI 助手：本地大模型、本地知识库、多模态图片识别、本地 API，并针对**第五代骁龙 8 至尊版（SM8850 / Adreno 840）**做了专属优化。

> 所有推理都在手机本地完成，不联网、不上云。模型权重、知识库、对话记录都留在设备上。

---

## 特性

| 需求 | 实现 |
|---|---|
| 本地知识库 | SQLite FTS5（BM25 关键词）+ 本地 embedding 向量，**RRF 混合检索** |
| GPU 加速（≥ OpenCL） | llama.cpp **Adreno OpenCL** 后端；CPU 后端自动回退 |
| CPU 推理 | llama.cpp CPU 后端（含 ARM KleidiAI 优化） |
| 图片识别 / 多模态 | libmtmd 多模态推理，图片作为对话输入 |
| MTP 投机解码 | 原生支持 Qwen3.8 / Nimbus / Ornith 等模型的 MTP 头，几乎零额外内存加速 |
| 最新模型支持 | Qwen3.8、Nimbus、Ornith 的 GGUF（见 `docs/MODELS.md`） |
| 本地 API | **OpenAI 兼容 HTTP**（/v1/chat/completions 等）+ **AIDL** 供其他 APK 调用 |
| 骁龙 8 Gen5 专属 | 见 `docs/SNAPDRAGON.md` |

---

## 架构

```
┌──────────────────────────────────────────────────────────┐
│  UI (Jetpack Compose)  MainActivity / MainScreen           │
│      │ 调用                                                │
│  MainViewModel  ── 聊天流、RAG 注入、知识库导入、API 开关  │
│      │                                                    │
│  AppServices（单例）                                      │
│   ├─ ModelRuntime  ── 模型加载 / 生成 / Mutex 串行化       │
│   ├─ Embedder      ── embedding 向量化                     │
│   ├─ KnowledgeDb   ── FTS5 + 向量存储                      │
│   ├─ HybridRetriever ── BM25 + 向量 RRF 融合              │
│   └─ LocalMindBackend (InferenceBackend)                  │
│            │                            │                 │
│      OpenAiRoutes + HttpServer    LocalMindApiService(AIDL)│
│            │                            │                 │
│   ════════ 本地 HTTP (127.0.0.1:8080) ══════ 跨 App 绑定 ═│
│            │                            │                 │
│  ┌─────────┴────────────────────────────┴──────────┐     │
│  │  liblocalmind.so  (JNI)                           │     │
│  │   lm_engine.cpp  ── MTP / 多模态 / OpenCL+CPU     │     │
│  │   llama.cpp (Adreno OpenCL · KleidiAI · mtmd)     │     │
│  └───────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

模型是**设备级共享资源**：本 App 的 UI 与外部 App 通过 AIDL 调用的是同一份已加载模型，不会各占一份内存。

---

## 目录结构

```
LocalMind/
├── app/src/main/
│   ├── cpp/                # native 层（已用 Linux 同级工具链验证编译通过）
│   │   ├── CMakeLists.txt  # 把 llama.cpp 作为子项目编译，开启 OpenCL/KleidiAI/mtmd
│   │   ├── lm_engine.h/.cpp# 推理引擎：生成、MTP、多模态、向量化
│   │   └── lm_jni.cpp      # JNI 绑定（11 个导出符号）
│   ├── aidl/               # ILocalMindApi / ILocalMindStreamCallback
│   ├── java/.../ai/
│   │   ├── engine/         # NativeEngine, ModelRuntime, SnapdragonTuner, AppServices
│   │   ├── rag/            # KnowledgeDb, Embedder, HybridRetriever, DocumentIngestor
│   │   ├── api/            # HttpServer(手写), OpenAiRoutes
│   │   └── ui/             # MainActivity, MainViewModel, MainScreen, Theme
│   └── res/                # 资源与清单
├── third_party/llama.cpp   # 由 scripts/fetch-llama-cpp.sh 拉取（不入库）
└── scripts/                # fetch-llama-cpp.sh, build_apk.sh
```

---

## 构建

> ⚠️ **沙箱构建限制（已实测）**：构建环境无法访问 Google 域名（dl.google.com / maven.google.com），国内 Android SDK 镜像仅缓存到 **NDK r21**（缺少 C++17 与 KleidiAI 支持），因此**无法在沙箱内产出 APK**。请在装有 Android SDK/NDK 的本地机器上构建。
>
> 不过，native 核心层（liblocalmind + llama.cpp，含 OpenCL 与多模态）**已在 Linux x86_64 用同级工具链真实编译通过**，11 个 JNI 符号导出验证正确——把出错风险在交付前消除了一大半。

前置（本机）：Android SDK（platform-35、build-tools 35.0.0）、NDK 27.0.12077973、CMake 3.22.1、JDK 17。

```bash
# 方式一：一键脚本
./scripts/build_apk.sh            # debug APK
./scripts/build_apk.sh release    # release APK（需自行配置签名）

# 方式二：Android Studio
# 直接打开本工程目录，等待同步后 Build → Build Bundle(s) / APK(s)
```

---

## CI 构建（GitHub Actions · 4 vCPU / 16 GB）

工程内置 `.github/workflows/build-release.yml`，在 GitHub 托管的 `ubuntu-latest`（4 vCPU / 16 GB RAM）上自动编译 **arm64-v8a release APK**，产物名为 `app-arm64-v8a-release.apk`。

触发方式：

- 推送 `main` 分支 → 自动编译并上传 APK 为 Action artifact。
- 推送 `v*` 标签（如 `v1.0.0`）→ 编译并把 APK 发布到 GitHub Release。
- 仓库 **Actions** 页手动 `workflow_dispatch`。

签名（不配置也能产出已签名、可安装的 APK）：

| 方式 | 操作 |
|---|---|
| **A · 稳定密钥（推荐）** | 仓库 `Settings → Secrets` 添加：`SIGNING_KEY`（`base64` 编码的 `.jks`）、`KEYSTORE_PASSWORD`、`KEY_ALIAS`、`KEY_PASSWORD` |
| **B · 无密钥** | 留空即可，工作流用 `keytool` 生成一次性密钥库，APK 照常签名可安装（但密钥每次不同，正式发布请用方式 A） |

工作流要点：仅编 `arm64-v8a`（`-Plocalmind.arm64Only=true`）、Gradle 内存钳到 3 GB / 并发 3（适配 16 GB）、SDK 与 Gradle 依赖均缓存以加速重跑。

---

## 使用流程

1. **加载模型**：「模型」页选 GGUF 文件（多模态模型额外选 mmproj，自投机可选 draft 模型），选性能档位，点「加载」。骁龙 8 Gen5 会被自动识别并应用专属参数。
2. **知识库**：先加载一个 embedding 模型（设置页配置路径），再到「知识库」导入 txt/md/pdf/图片，自动切分+向量化+BM25 建库。对话开启「使用知识库」即自动检索注入。
3. **对话 / 图片**：「对话」页直接聊；点「图片」附图后发送即可多模态问答。
4. **本地 API**：「API」页开启 HTTP 服务，其他 App / 脚本即可用 OpenAI 客户端连 `http://127.0.0.1:8080`。外部 APK 也可绑定 `LocalMindApiService` 走 AIDL。

---

## 已知限制

- APK 需在本机 Android Studio / 命令行构建（见上文沙箱限制）。
- Qwen3.8-27B / Ornith-31B 等大模型对内存要求高，仅 24GB 机型勉强可跑，建议用 Q4 量化与 4B~9B 级别。
- 模型 GGUF 需从官方渠道获取（HuggingFace 等），本仓库不含权重文件。
- NPU（Hexagon QNN）后端已预留 `GGML_QNN` 开关，但需单独下载 Qualcomm QNN SDK 并自行开启，本文档默认走 OpenCL+CPU。

详见 `docs/MODELS.md`、`docs/API.md`、`docs/SNAPDRAGON.md`。
