# 支持的模型

LocalMind 基于 llama.cpp，只要模型有 **GGUF** 格式，就能加载——无需为每种架构单独写算子。下面是你点名要支持的几个家族，以及在本 App 里的配置方式。

> 权重文件需自行从官方渠道下载（HuggingFace GGUF、官方发布页等），本仓库不含任何 `.gguf`。

---

## 1. Qwen3.8 系列

- **Qwen3.8-27B Dense**：多模态（文本+视觉），采用 Gated DeltaNet + Gated Attention 混合架构。Q4 量化约 16GB，需 24GB 内存机型勉强运行，会挤压 KV cache。
- **Qwen3.8-Flash-Next**（MoE，Qwen4 架构预览）：MoE 需把全部专家载入内存，手机端不现实，**不建议在手机上跑**。

**加载方式**

| 项 | 配置 |
|---|---|
| 主模型 | 选 `qwen3.8-27b-...-q4_k_m.gguf` |
| 视觉 | 选对应 `mmproj-*.gguf`（多模态必需） |
| 投机解码 | `MTP`（默认开，模型自带 MTP 头） |
| 建议量化 | Q4_K_M / IQ4_XS（内存紧张时） |
| 建议档位 | 长上下文场景用「长上下文」，纯聊天用「均衡」 |

> 模板无需手动设置：Qwen3.8 自带 chat template，引擎按模型内部模板渲染，避免写死 ChatML 导致特殊 token 错乱。

---

## 2. Nimbus

- **Nimbus 2B / 4B / 9B v2.1**：官方提供 GGUF（BF16 + Q5_K_M），纯文本 coding 模型。9B 的 Q5_K_M 约 6GB，骁龙 8 Gen5 可流畅运行。

**加载方式**

| 项 | 配置 |
|---|---|
| 主模型 | 选 `nimbus-9b-...-q5_k_m.gguf`（或 2B/4B 轻量档） |
| 视觉 | 不需要（纯文本） |
| 投机解码 | `MTP`；若你额外下载了小 draft 模型，可改用 `DRAFT_MODEL` 指定 draft 路径 |
| 建议档位 | 「极速」追求低延迟代码补全 |

---

## 3. Ornith

- **Ornith-1.0**：MIT 许可，提供 9B Dense / 31B Dense / 35B MoE / 397B 多个规格。
  - **9B Dense**：手机现实选择，Q4 约 5–6GB。
  - **31B Dense**：Q4 约 17GB 起，超出多数机型内存，不建议。
  - **35B MoE / 397B**：服务器级。

**加载方式**

| 项 | 配置 |
|---|---|
| 主模型 | 选 `ornith-9b-...-q4_k_m.gguf` |
| 视觉 | 视官方是否提供 mmproj；有则配置 |
| 投机解码 | `MTP` |
| 建议档位 | 「均衡」或「省内存」 |

---

## 通用配置建议

- **量化选择**：内存 ≤12GB 优先 Q4_K_M；≥16GB 可上 Q5/Q6；24GB 旗舰可尝试 Q8。
- **上下文长度**：默认按设备内存预算自动设定（见 SnapdragonTuner）。KV cache 精度可在「省内存 / 长上下文」档位自动降到 Q8_0 / Q4_0。
- **多模态**：带视觉的模型务必配 `mmproj`，否则图片输入会被忽略。
- **embedding 模型**：知识库向量检索需要一个 embedding GGUF（如 Qwen3-Embedding 系列），在「设置」页配置路径后加载，再导入文档即可获得语义检索。

---

## 下载来源（建议）

- HuggingFace：搜索 `Qwen`、`Nimbus`、`Ornith` 官方 / 社区 GGUF 仓库。
- 视觉投影：与对应模型同仓库的 `mmproj-*.gguf`。
- 模型尺寸量力而行：骁龙 8 Gen5 实际可用模型预算约 14GB（受 Android 单进程内存上限约束）。
