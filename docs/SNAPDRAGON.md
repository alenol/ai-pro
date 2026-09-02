# 第五代骁龙 8 至尊版（SM8850 / Adreno 840）专属优化

LocalMind 的 native 层在 `lm_engine.cpp` 与 `SnapdragonTuner.kt` 中针对该平台做了专门调校。下面说明每一条决策的依据。

---

## 1. GPU 后端：Adreno OpenCL

- 使用 llama.cpp 的 **Adreno OpenCL** 后端（`GGML_OPENCL=ON` + `GGML_OPENCL_USE_ADRENO_KERNELS=ON`），社区与高通官方均有维护，IWOCL 2026 有专门论文。
- **关键**：把 OpenCL 内核源码编进 `.so`（`GGML_OPENCL_EMBED_KERNELS=ON`）。Android 文件系统里没有内核 `.cl` 文件，运行时无法像桌面那样从磁盘加载，必须从二进制内取。
- CPU 回退：探测不到 OpenCL 设备时（`nativeProbe` 返回 `opencl_available=false`）自动纯 CPU 推理。

---

## 2. CPU 后端：ARM KleidiAI

- arm64-v8a 构建开启 `GGML_CPU_KLEIDIAI`，利用 Arm KleidiAI 的矩阵乘内核，在必须落到 CPU 的层（如输出层、部分注意力）上显著提速。
- 线程数由 `SnapdragonTuner` 控制：生成阶段 **4–6 线程**，而非开满 8 核——SM8850 是 2 超大核 + 6 性能核的 Oryon 布局，开满会把任务摊到正在降频的核心上，反而更慢更热。prompt 处理阶段可短暂用满核心。

---

## 3. MTP 投机解码（手机端首选）

- Qwen3.8 / Nimbus / Ornith 等模型自带 **MTP 多头预测头**。开启 `load_mtp=true` 后，draft context 复用主模型权重，**几乎零额外内存**就能加速生成。
- **坑**：llama.cpp 中 `load_mtp` 默认是 `false`——不显式打开，MTP 权重不会加载，加速是空的。LocalMind 默认开启。
- 接受率（`acceptRate`）会在生成统计里返回，可直接看到 MTP 是否真的在加速。

---

## 4. KV cache 精度 ↔ 上下文长度

手机上 GPU 与 CPU 共享同一块 LPDDR5X，没有独立显存。KV cache 精度直接决定能开多长上下文：

| 精度 | 每 token 占用（典型 9B/36层/4 KV head） |
|---|---|
| F16  | ~72 KB |
| Q8_0 | ~37 KB |
| Q4_0 | ~19 KB |

默认按设备内存预算自动选择：`长上下文`档位用 Q8_0，`省内存`档用 Q4_0，其余优先 F16。

---

## 5. 层数卸载策略

- 全卸载（`n_gpu_layers=-1`）在内存充裕时最快。但全部压到 Adreno 后，KV cache 也被挤进同一块内存，上下文一长就 OOM。
- 因此 `SnapdragonTuner.buildConfig` 在预算紧张时会把输出层/词嵌入留在 CPU（见代码里 `recommendGpuLayers` 的注释）。

---

## 6. 内存预算

`DeviceProfile.modelBudgetMb` 按物理内存分级估算可用模型预算（8GB→35%、12GB→45%、16GB→55%、24GB+→65%），并再压一道 Android 单进程上限（≤14GB）。UI 的「骁龙专属调优建议」会实时显示：

> 模型 X MB + KV Y MB = Z MB / 可用 W MB
> ⚠ 预计超出预算，建议降低上下文或改用 Q8_0/Q4_0

---

## 7. NPU（Hexagon QNN）预留

- `CMakeLists.txt` 保留 `GGML_QNN` 开关（默认关闭）。开启需单独下载 Qualcomm QNN SDK 并接受许可，构建复杂度与失败率明显上升，且算子覆盖不全需回退。
- 当前默认走 OpenCL + CPU，全开源、构建链路干净，骁龙 8 Gen5 上 7B~9B 模型通常可达 15–25 tok/s。

---

## 性能预期（参考）

在第五代骁龙 8 至尊版（Adreno 840）上，OpenCL 后端 + MTP：

- **4B 级**（如 Nimbus-4B）：30–45 tok/s
- **9B 级**（Nimbus-9B / Ornith-9B）：15–25 tok/s
- **27B 级**（Qwen3.8-27B Q4）：受内存与带宽限制，个位数到十余 tok/s，且上下文较短

> 实际数值取决于量化、上下文长度、是否开 MTP 与后台负载。以上为同平台公开实测的粗略区间，非本工程实测承诺。
