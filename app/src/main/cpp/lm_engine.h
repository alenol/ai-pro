#pragma once

// LocalMind 原生推理引擎。
//
// 这一层是纯 C++，不依赖任何 Android API，因此可以在开发机上直接编译验证。
// JNI 绑定单独放在 lm_jni.cpp。
//
// 能力：
//   - 文本生成（含 MTP / 独立 draft 模型 / ngram 三种投机解码）
//   - 多模态图片输入（libmtmd）
//   - 文本向量化（本地知识库检索用）
//   - OpenCL(Adreno GPU) / CPU 混合后端

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"

namespace localmind {

// ---------------------------------------------------------------------------
// 投机解码模式
// ---------------------------------------------------------------------------
enum class SpecMode {
    OFF = 0,
    // 模型自带 MTP 头（Qwen3.5 / Qwen3.8 / step35 系列）。
    // 不需要额外的 draft 模型文件，draft context 直接复用主模型权重，
    // 因此在手机上几乎零额外内存开销，是移动端首选。
    MTP = 1,
    // 独立的小 draft 模型（如 Nimbus-2B 给 9B 做 draft）。
    // 需要额外加载一份权重，内存成本明显。
    DRAFT_MODEL = 2,
    // ngram 匹配，纯 CPU、零额外内存。
    // 对代码补全、模板化文本、重复片段效果很好，通用对话增益有限。
    NGRAM = 3,
};

// ---------------------------------------------------------------------------
// KV cache 精度。手机内存紧张时降精度可以换来翻倍的上下文长度。
// ---------------------------------------------------------------------------
enum class CacheType {
    F16 = 0,
    Q8_0 = 1,
    Q4_0 = 2,
};

// ---------------------------------------------------------------------------
// 模型加载配置
// ---------------------------------------------------------------------------
struct ModelConfig {
    std::string model_path;   // 主模型 GGUF
    std::string mmproj_path;  // 视觉投影 GGUF，留空表示纯文本模型
    std::string draft_path;   // 仅 SpecMode::DRAFT_MODEL 需要

    int32_t n_ctx           = 8192;
    int32_t n_batch         = 2048;  // 逻辑批大小（prompt 处理）
    int32_t n_ubatch        = 512;   // 物理批大小（决定峰值显存/内存）

    // 卸载到 GPU 的层数。-1 表示全部。
    // 手机 GPU 显存与内存共享，层数过多会导致 KV cache 被压缩，
    // 具体推荐值见 SnapdragonTuner。
    int32_t n_gpu_layers    = -1;

    int32_t n_threads       = 4;     // 生成阶段 CPU 线程
    int32_t n_threads_batch = 0;     // 0 表示自动（= n_threads）

    bool flash_attn = true;   // 骁龙上强烈建议开启

    // 是否加载模型自带的 MTP 权重。
    // llama.cpp 中该选项默认为 false：MTP 头会占用额外内存，
    // 纯生成场景没必要加载。开启 SpecMode::MTP 时必须为 true。
    bool load_mtp   = true;

    CacheType cache_type_k = CacheType::F16;
    CacheType cache_type_v = CacheType::F16;

    SpecMode spec_mode = SpecMode::MTP;
    int32_t  spec_n_max = 16;    // 每次最多起草多少 token
    int32_t  spec_n_min = 4;
    float    spec_p_min = 0.75f; // 低于该接受率的草稿会被降权
};

// ---------------------------------------------------------------------------
// 采样参数
// ---------------------------------------------------------------------------
struct SamplerConfig {
    float    temp          = 0.7f;
    float    top_p         = 0.9f;
    float    top_k         = 40.0f;
    float    min_p         = 0.05f;
    float    repeat_penalty = 1.1f;
    int32_t  repeat_last_n = 64;
    int32_t  n_predict     = -1;   // -1 = 直到 EOS 或上下文耗尽
    uint32_t seed          = 0;    // 0 = 每轮随机
};

// ---------------------------------------------------------------------------
// 图片输入：必须是紧凑的 RGB 数据（nx * ny * 3 字节），无行填充。
// 由 Kotlin 侧解码 Bitmap 后转换而来。
// ---------------------------------------------------------------------------
struct ImageInput {
    uint32_t              nx = 0;
    uint32_t              ny = 0;
    std::vector<uint8_t>  rgb;
};

// ---------------------------------------------------------------------------
// 一次生成的结果统计
// ---------------------------------------------------------------------------
struct GenStats {
    int32_t n_prompt_tokens = 0;
    int32_t n_gen_tokens    = 0;

    double  t_prompt_ms = 0.0;
    double  t_gen_ms    = 0.0;

    double  prompt_tps  = 0.0;
    double  gen_tps     = 0.0;

    // 投机解码相关
    int32_t n_drafted    = 0;
    int32_t n_accepted   = 0;
    double  accept_rate  = 0.0;  // 接受率，衡量 MTP 是否真的在加速

    std::string stop_reason;  // "eos" | "length" | "cancelled" | "error"
};

// 逐 token 回调。返回 false 表示中止生成。
using TokenCallback = std::function<bool(const std::string & piece)>;

// ---------------------------------------------------------------------------
// 对话消息
// ---------------------------------------------------------------------------
struct ChatMsg {
    std::string role;     // "system" | "user" | "assistant" | "tool"
    std::string content;
};

// ---------------------------------------------------------------------------
// 生成引擎（一个实例 = 一个已加载的主模型）
// ---------------------------------------------------------------------------
class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine &)            = delete;
    Engine & operator=(const Engine &) = delete;

    bool load(const ModelConfig & cfg, std::string & err);
    void unload();
    bool is_loaded() const;

    // 流式生成。images 非空时 prompt 中需含图片占位符（见 lm_jni.cpp）。
    bool generate(const std::string & prompt,
                  const std::vector<ImageInput> & images,
                  const SamplerConfig & sp,
                  TokenCallback cb,
                  std::string & err);

    // 中断当前生成（线程安全，可在回调线程外调用）
    void cancel();

    const GenStats & stats() const { return stats_; }

    int32_t     n_ctx() const;
    int32_t     n_embd() const;
    bool        supports_vision() const;
    std::string model_desc() const;   // 供 UI 展示的架构/参数摘要
    std::string backend_desc() const; // 实际生效的后端（OpenCL/CPU）

    // 按模型自带的 chat template 渲染对话。
    //
    // 不要在上层硬编码 ChatML：Qwen3.8、Nimbus、Ornith 的模板并不相同，
    // 写死会导致特殊 token 错乱、回答质量骤降。模型 GGUF 内自带模板，
    // 交给 llama.cpp 解析是唯一稳妥的做法。
    std::string apply_template(const std::vector<ChatMsg> & msgs,
                               bool add_generation_prompt = true) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    GenStats              stats_;
    std::atomic<bool>     cancelled_{false};
};

// ---------------------------------------------------------------------------
// 向量化器（本地知识库检索用）
//
// 独立于生成引擎，因为 embedding 模型通常比主模型小得多
// （例如 Qwen3-Embedding-0.6B），且需要不同的 pooling 与批处理策略。
// ---------------------------------------------------------------------------
class Embedder {
public:
    Embedder();
    ~Embedder();

    Embedder(const Embedder &)            = delete;
    Embedder & operator=(const Embedder &) = delete;

    bool load(const std::string & model_path,
              int32_t n_ctx,
              int32_t n_gpu_layers,
              int32_t n_threads,
              std::string & err);
    void unload();
    bool is_loaded() const;

    // 批量文本 -> L2 归一化向量。已按模型自身 batch 上限自动切分。
    bool embed(const std::vector<std::string> & texts,
               std::vector<std::vector<float>> & out,
               std::string & err);

    int32_t n_embd() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// ---------------------------------------------------------------------------
// 运行时探测（用于设置页展示与自动调优）
// ---------------------------------------------------------------------------
struct RuntimeInfo {
    std::string soc_model;      // 例如 "SM8850"
    std::string gpu_name;       // 例如 "Adreno (TM) 840"
    int32_t     n_cores        = 0;
    int64_t     total_ram_mb   = 0;
    bool        opencl_available = false;
    std::string opencl_version;
};

RuntimeInfo probe_runtime();

}  // namespace localmind
