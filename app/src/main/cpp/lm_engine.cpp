// LocalMind 原生推理引擎实现。
//
// 设计原则：
//   1. 这一层不碰任何 Android API，可在开发机上独立编译验证。
//   2. 复用 llama.cpp 官方的 common / mtmd / speculative 组件，
//      不自己重写采样器、投机解码和多模态批处理逻辑 —— 这些是
//      正确性风险最高的部分，自己写等于重新踩一遍上游踩过的坑。

#include "lm_engine.h"
#include "lm_log.h"

#include "llama.h"
#include "ggml-backend.h"

#include "common.h"
#include "chat.h"
#include "sampling.h"
#include "speculative.h"

#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif

namespace localmind {

using Clock = std::chrono::steady_clock;

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
               Clock::now().time_since_epoch()).count();
}

static ggml_type to_ggml_type(CacheType t) {
    switch (t) {
        case CacheType::Q8_0: return GGML_TYPE_Q8_0;
        case CacheType::Q4_0: return GGML_TYPE_Q4_0;
        case CacheType::F16:
        default:              return GGML_TYPE_F16;
    }
}

// ===========================================================================
// Engine
// ===========================================================================

struct Engine::Impl {
    ModelConfig cfg;

    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    mtmd_context  * mtmd  = nullptr;

    // 投机解码
    llama_model       * model_dft = nullptr;  // 仅 DRAFT_MODEL 模式
    llama_context     * ctx_dft   = nullptr;  // MTP 模式与主模型共享权重
    common_speculative_ptr spec;              // nullptr 表示未启用

    // 采样器在每次生成开始时重建（保证 repeat penalty 等状态干净）
    common_sampler * smpl = nullptr;

    llama_batch batch_gen;  // 生成阶段复用的 batch，避免每轮 malloc
    bool        batch_gen_ready = false;

    // 对话模板缓存。解析 Jinja 模板有开销，只做一次。
    mutable common_chat_templates_ptr chat_tmpls;
};

Engine::Engine()  : impl(new Impl) {}
Engine::~Engine() { unload(); }

bool Engine::is_loaded() const { return impl && impl->ctx != nullptr; }

int32_t Engine::n_ctx() const {
    return impl && impl->ctx ? llama_n_ctx(impl->ctx) : 0;
}

int32_t Engine::n_embd() const {
    return impl && impl->model ? llama_model_n_embd(impl->model) : 0;
}

bool Engine::supports_vision() const { return impl && impl->mtmd != nullptr; }

std::string Engine::model_desc() const {
    if (!impl->model) return "";
    char buf[512];
    llama_model_desc(impl->model, buf, sizeof(buf));
    return std::string(buf);
}

std::string Engine::backend_desc() const {
    if (!impl->ctx) return "";
    std::string s;
    // 遍历上下文实际使用的后端缓冲类型，反映真实生效的后端
    for (uint32_t i = 0; i < 32; i++) {
        // llama 不直接暴露后端列表，这里通过全局设备注册表推断
        break;
    }
    // 用设备注册表判断 OpenCL 是否参与
    const size_t n = ggml_backend_dev_count();
    bool has_cl = false;
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            has_cl = true;
        }
    }
    s += has_cl ? "OpenCL(GPU)+CPU" : "CPU";
    return s;
}

std::string Engine::apply_template(const std::vector<ChatMsg> & msgs,
                                   bool add_generation_prompt) const {
    if (!impl->model) return {};

    if (!impl->chat_tmpls) {
        impl->chat_tmpls = common_chat_templates_init(impl->model, /*override=*/ nullptr);
        if (!impl->chat_tmpls) {
            LM_LOGW("模型未提供对话模板，将按原始文本发送");
            return {};
        }
    }

    common_chat_templates_inputs inputs;
    inputs.messages.reserve(msgs.size());
    for (const auto & m : msgs) {
        common_chat_msg msg;
        msg.role    = m.role;
        msg.content = m.content;
        inputs.messages.push_back(std::move(msg));
    }
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.use_jinja             = true;

    common_chat_params params = common_chat_templates_apply(impl->chat_tmpls.get(), inputs);
    return params.prompt;
}

void Engine::cancel() { cancelled_.store(true); }

// ---------------------------------------------------------------------------
// 加载
// ---------------------------------------------------------------------------
bool Engine::load(const ModelConfig & cfg, std::string & err) {
    unload();

    impl->cfg = cfg;

    if (cfg.model_path.empty()) {
        err = "模型路径为空";
        return false;
    }

    llama_backend_init();

    // ---- 主模型 ----
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    // 上游默认不加载 MTP 权重（省内存），用 MTP 时必须显式打开，
    // 否则 llama_model_n_layer_nextn() 返回 0，投机解码会被静默关闭。
    mparams.load_mtp     = cfg.load_mtp && cfg.spec_mode == SpecMode::MTP;

    const double t0 = now_ms();
    impl->model = llama_model_load_from_file(cfg.model_path.c_str(), mparams);
    if (!impl->model) {
        err = "加载模型失败: " + cfg.model_path;
        return false;
    }
    LM_LOGI("模型加载完成 %.0f ms", now_ms() - t0);

    // ---- 上下文 ----
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = cfg.n_ctx;
    cparams.n_batch         = cfg.n_batch;
    cparams.n_ubatch        = cfg.n_ubatch;
    cparams.n_threads       = cfg.n_threads;
    cparams.n_threads_batch = cfg.n_threads_batch > 0 ? cfg.n_threads_batch : cfg.n_threads;
    cparams.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                             : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.type_k          = to_ggml_type(cfg.cache_type_k);
    cparams.type_v          = to_ggml_type(cfg.cache_type_v);
    cparams.no_perf         = false;

    // n_ubatch 不能超过 n_batch
    if (cparams.n_ubatch > cparams.n_batch) cparams.n_ubatch = cparams.n_batch;

    impl->ctx = llama_init_from_model(impl->model, cparams);
    if (!impl->ctx) {
        err = "创建推理上下文失败（显存/内存不足？）";
        unload();
        return false;
    }

    // ---- 视觉投影（多模态）----
    if (!cfg.mmproj_path.empty()) {
        mtmd_context_params mctx = mtmd_context_params_default();
        mctx.use_gpu         = cfg.n_gpu_layers != 0;
        mctx.flash_attn_type = cfg.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                              : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        mctx.n_threads       = cfg.n_threads;
        mctx.warmup          = false;  // 冷启动更快，首图稍慢

        impl->mtmd = mtmd_init_from_file(cfg.mmproj_path.c_str(), impl->model, mctx);
        if (!impl->mtmd) {
            LM_LOGW("视觉投影加载失败，将退化为纯文本模式: %s", cfg.mmproj_path.c_str());
            // 非致命：允许纯文本继续工作
        } else {
            LM_LOGI("多模态已启用 (vision=%d audio=%d)",
                    (int) mtmd_support_vision(impl->mtmd),
                    (int) mtmd_support_audio(impl->mtmd));
        }
    }

    // ---- 投机解码 ----
    if (cfg.spec_mode == SpecMode::MTP) {
        const int32_t n_nextn = llama_model_n_layer_nextn(impl->model);
        if (n_nextn <= 0) {
            LM_LOGW("该模型不含 MTP 头（n_nextn=%d），已自动关闭投机解码", n_nextn);
        } else {
            llama_context_params cp  = cparams;
            cp.ctx_type  = LLAMA_CONTEXT_TYPE_MTP;
            cp.n_ctx     = llama_n_ctx(impl->ctx);
            cp.n_rs_seq  = 0;      // MTP 不需要回滚快照
            cp.ctx_other = impl->ctx;

            impl->ctx_dft = llama_init_from_model(impl->model, cp);
            if (!impl->ctx_dft) {
                LM_LOGW("MTP 上下文创建失败，已自动关闭投机解码");
            } else {
                LM_LOGI("MTP 已启用 (n_nextn=%d)", n_nextn);
            }
        }
    } else if (cfg.spec_mode == SpecMode::DRAFT_MODEL) {
        if (cfg.draft_path.empty()) {
            LM_LOGW("DRAFT_MODEL 模式未提供 draft 模型路径，已关闭投机解码");
        } else {
            impl->model_dft = llama_model_load_from_file(cfg.draft_path.c_str(), mparams);
            if (!impl->model_dft) {
                LM_LOGW("draft 模型加载失败，已关闭投机解码");
            } else {
                llama_context_params cp  = cparams;
                cp.ctx_type  = LLAMA_CONTEXT_TYPE_DEFAULT;
                cp.n_ctx     = llama_n_ctx(impl->ctx);
                cp.n_rs_seq  = 0;
                cp.ctx_other = impl->ctx;

                impl->ctx_dft = llama_init_from_model(impl->model_dft, cp);
                if (!impl->ctx_dft) {
                    LM_LOGW("draft 上下文创建失败，已关闭投机解码");
                    llama_model_free(impl->model_dft);
                    impl->model_dft = nullptr;
                }
            }
        }
    }

    // 组装 speculative 实例
    if (impl->ctx_dft) {
        common_params_speculative sparams;
        sparams.draft.n_max = cfg.spec_n_max;
        sparams.draft.n_min = cfg.spec_n_min;
        sparams.draft.p_min = cfg.spec_p_min;
        sparams.draft.ctx_tgt = impl->ctx;
        sparams.draft.ctx_dft = impl->ctx_dft;
        sparams.draft.cache_type_k = to_ggml_type(cfg.cache_type_k);
        sparams.draft.cache_type_v = to_ggml_type(cfg.cache_type_v);

        switch (cfg.spec_mode) {
            case SpecMode::MTP:
                sparams.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
                break;
            case SpecMode::DRAFT_MODEL:
                sparams.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE);
                break;
            case SpecMode::NGRAM:
                sparams.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
                break;
            default:
                break;
        }

        if (!sparams.types.empty()) {
            impl->spec.reset(common_speculative_init(sparams, /*n_seq=*/ 1));
            if (!impl->spec) {
                LM_LOGW("投机解码初始化失败，已降级为普通解码");
            }
        }
    } else if (cfg.spec_mode == SpecMode::NGRAM) {
        // ngram 不需要 draft 模型，但同样需要 spec 实例
        common_params_speculative sparams;
        sparams.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        sparams.draft.n_max = cfg.spec_n_max;
        sparams.draft.n_min = cfg.spec_n_min;
        sparams.draft.p_min = cfg.spec_p_min;
        impl->spec.reset(common_speculative_init(sparams, /*n_seq=*/ 1));
    }

    // ---- 生成阶段复用的 batch ----
    const int32_t cap = 1 + std::max(0, cfg.spec_n_max);
    impl->batch_gen  = llama_batch_init(cap, /*embd=*/ 0, /*n_seq_max=*/ 1);
    impl->batch_gen_ready = true;

    LM_LOGI("引擎就绪: ctx=%d batch=%d ubatch=%d backend=%s",
            llama_n_ctx(impl->ctx), cfg.n_batch, cfg.n_ubatch, backend_desc().c_str());
    return true;
}

void Engine::unload() {
    if (!impl) return;

    if (impl->batch_gen_ready) {
        llama_batch_free(impl->batch_gen);
        impl->batch_gen_ready = false;
    }

    impl->spec.reset();
    impl->chat_tmpls.reset();

    if (impl->smpl) { common_sampler_free(impl->smpl); impl->smpl = nullptr; }
    if (impl->ctx_dft)  { llama_free(impl->ctx_dft);   impl->ctx_dft = nullptr; }
    if (impl->mtmd)     { mtmd_free(impl->mtmd);       impl->mtmd = nullptr; }
    if (impl->ctx)      { llama_free(impl->ctx);       impl->ctx = nullptr; }
    if (impl->model_dft){ llama_model_free(impl->model_dft); impl->model_dft = nullptr; }
    if (impl->model)    { llama_model_free(impl->model);     impl->model = nullptr; }

    llama_backend_free();
}

// ---------------------------------------------------------------------------
// 分词
// ---------------------------------------------------------------------------
static std::vector<llama_token> tokenize_text(const llama_model * model,
                                              const std::string & text,
                                              bool add_special,
                                              bool parse_special) {
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // 先按 UTF-8 字节上界分配，再收缩
    const int32_t n_max = (int32_t) text.size() + 2 * (int32_t) std::count_if(
                              text.begin(), text.end(),
                              [](unsigned char c) { return (c & 0xC0) == 0x80; }) + 4;

    std::vector<llama_token> out(n_max);
    int32_t n = llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                               out.data(), (int32_t) out.size(),
                               add_special, parse_special);
    if (n < 0) {
        out.resize((size_t) (-n));
        n = llama_tokenize(vocab, text.data(), (int32_t) text.size(),
                           out.data(), (int32_t) out.size(),
                           add_special, parse_special);
    }
    if (n < 0) return {};
    out.resize((size_t) n);
    return out;
}

// ---------------------------------------------------------------------------
// 生成
// ---------------------------------------------------------------------------
bool Engine::generate(const std::string & prompt,
                      const std::vector<ImageInput> & images,
                      const SamplerConfig & sp,
                      TokenCallback cb,
                      std::string & err) {
    Impl & I = *impl;

    if (!I.ctx) { err = "模型未加载"; return false; }

    LM_LOGI("generate begin: prompt_len=%zu gpu_layers=%d flash=%d spec=%d ctx=%d ubatch=%d",
            prompt.size(), I.cfg.n_gpu_layers, (int) I.cfg.flash_attn,
            (int) I.cfg.spec_mode, I.cfg.n_ctx, I.cfg.n_ubatch);

    cancelled_.store(false);
    stats_ = GenStats{};

    llama_model   * model = I.model;
    llama_context * ctx   = I.ctx;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const llama_token eos = llama_vocab_eos(vocab);

    // 清空 KV cache（每次生成独立，避免受上一轮污染）
    if (llama_memory_t mem = llama_get_memory(ctx)) {
        llama_memory_seq_rm(mem, /*seq_id=*/ 0, /*p0=*/ 0, /*p1=*/ -1);
    }

    // ---- 采样器 ----
    common_params_sampling sparams;
    sparams.temp           = sp.temp;
    sparams.top_p          = sp.top_p;
    sparams.top_k          = (int32_t) sp.top_k;
    sparams.min_p          = sp.min_p;
    sparams.penalty_repeat = sp.repeat_penalty;
    sparams.penalty_last_n = sp.repeat_last_n;
    // 顺序即采样链的执行顺序，温度必须放在最后。
    sparams.samplers = {
        COMMON_SAMPLER_TYPE_PENALTIES,
        COMMON_SAMPLER_TYPE_TOP_K,
        COMMON_SAMPLER_TYPE_TOP_P,
        COMMON_SAMPLER_TYPE_MIN_P,
        COMMON_SAMPLER_TYPE_TEMPERATURE,
    };
    if (sp.seed != 0) sparams.seed = sp.seed;

    if (I.smpl) common_sampler_free(I.smpl);
    I.smpl = common_sampler_init(model, sparams);
    if (!I.smpl) { err = "采样器初始化失败"; return false; }

    // ------------------------------------------------------------------
    // 1) 处理输入
    // ------------------------------------------------------------------
    const bool multimodal = !images.empty() && I.mtmd != nullptr;

    std::vector<llama_token> prompt_tokens;
    mtmd_input_chunks * chunks = nullptr;

    const double t_prompt_0 = now_ms();

    if (multimodal) {
        // 图片走 mtmd 编码。
        //
        // 注意：投机解码在此路径下自动关闭。
        // 原因：llama.cpp 的 spec 机制以「文本 token 序列」为单位起草，
        // 而图片 chunk 是 embedding 输入，没有对应的 token 序列。
        // 强行让二者共存会破坏草稿与 KV cache 的位置对应关系。
        // 视觉问答的瓶颈本来也在图像编码而非自回归解码，收益有限。
        if (I.spec) {
            LM_LOGD("多模态输入：本次生成自动关闭投机解码");
        }

        chunks = mtmd_input_chunks_init();
        std::vector<mtmd_bitmap *> bms;
        bms.reserve(images.size());
        for (const auto & img : images) {
            if (img.rgb.size() < (size_t) img.nx * img.ny * 3) {
                err = "图片数据不完整";
                mtmd_input_chunks_free(chunks);
                return false;
            }
            bms.push_back(mtmd_bitmap_init(img.nx, img.ny, img.rgb.data()));
        }

        mtmd_input_text tin{
            /* .text          = */ prompt.c_str(),
            /* .text_len      = */ prompt.size(),
            /* .add_special   = */ true,
            /* .parse_special = */ true,
        };

        // mtmd_tokenize 要求 const 指针数组
        std::vector<const mtmd_bitmap *> cbms(bms.begin(), bms.end());
        const int32_t rc = mtmd_tokenize(I.mtmd, chunks, &tin,
                                         cbms.data(), cbms.size());
        for (auto * b : bms) mtmd_bitmap_free(b);

        if (rc != 0) {
            err = "图片 tokenize 失败，code=" + std::to_string(rc);
            mtmd_input_chunks_free(chunks);
            return false;
        }

        // 逐 chunk 编码进 KV cache
        llama_pos n_past = 0;
        const size_t n_chunks = mtmd_input_chunks_size(chunks);
        for (size_t c = 0; c < n_chunks; c++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, c);
            llama_pos new_n_past = 0;
            if (mtmd_helper_eval_chunk_single(I.mtmd, ctx, chunk, n_past,
                                              /*seq_id=*/ 0, I.cfg.n_batch,
                                              /*logits_last=*/ false,
                                              &new_n_past) != 0) {
                err = "多模态编码失败";
                mtmd_input_chunks_free(chunks);
                return false;
            }
            n_past = new_n_past;
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &n);
                prompt_tokens.insert(prompt_tokens.end(), toks, toks + n);
            }
        }
        stats_.n_prompt_tokens = (int32_t) n_past;
    } else {
        prompt_tokens = tokenize_text(model, prompt, /*add_special=*/ true,
                                      /*parse_special=*/ true);
        if (prompt_tokens.empty()) {
            err = "prompt 分词结果为空";
            return false;
        }
        stats_.n_prompt_tokens = (int32_t) prompt_tokens.size();

        // 分批送入 prompt。只有最后一批的最后一个 token 需要 logits。
        const size_t total = prompt_tokens.size();
        const size_t step  = (size_t) std::max(1, I.cfg.n_batch);

        for (size_t off = 0; off < total; off += step) {
            const size_t n = std::min(step, total - off);
            const bool   last = (off + n == total);

            llama_batch batch = llama_batch_init((int32_t) n, 0, 1);
            for (size_t k = 0; k < n; k++) {
                batch.token[k]    = prompt_tokens[off + k];
                batch.pos[k]      = (llama_pos) (off + k);
                batch.n_seq_id[k] = 1;
                batch.seq_id[k][0] = 0;
                batch.logits[k]   = (last && k + 1 == n) ? 1 : 0;
            }

            if (llama_decode(ctx, batch) != 0) {
                err = "prompt 解码失败（上下文或批大小超限）";
                llama_batch_free(batch);
                return false;
            }
            if (I.spec) common_speculative_process(I.spec.get(), batch);
            llama_batch_free(batch);

            if (cancelled_.load()) {
                stats_.stop_reason = "cancelled";
                return true;
            }
        }

        if (I.spec) common_speculative_begin(I.spec.get(), 0, prompt_tokens);
    }

    stats_.t_prompt_ms = now_ms() - t_prompt_0;
    if (stats_.t_prompt_ms > 0) {
        stats_.prompt_tps = stats_.n_prompt_tokens / (stats_.t_prompt_ms / 1000.0);
    }

    // ------------------------------------------------------------------
    // 2) 自回归生成
    // ------------------------------------------------------------------
    const double t_gen_0 = now_ms();

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // 多模态时禁用投机解码（见上文注释）
    common_speculative * spec = multimodal ? nullptr : I.spec.get();

    // 首 token：直接采样（投机解码需要有一个已确认的 token 作为起草基准）
    llama_token id = common_sampler_sample(I.smpl, ctx, /*idx=*/ -1);
    common_sampler_accept(I.smpl, id, /*is_generated=*/ true);

    llama_pos n_past = (llama_pos) stats_.n_prompt_tokens;
    std::string stop_reason;

    const int32_t n_predict_max = sp.n_predict < 0
                                      ? (int32_t) (llama_n_ctx(ctx) - stats_.n_prompt_tokens - 1)
                                      : sp.n_predict;

    int32_t n_gen = 0;
    bool    ok    = true;

    // 先吐出首 token
    if (id == eos || id >= n_vocab) {
        stop_reason = "eos";
        ok = true;
    } else {
        std::string piece = common_token_to_piece(ctx, id, /*special=*/ false);
        if (!piece.empty() && cb && !cb(piece)) { cancelled_.store(true); }
        n_gen++;
        n_past++;
        prompt_tokens.push_back(id);
    }

    // 主循环
    while (ok && stop_reason.empty() && n_gen < n_predict_max) {
        if (cancelled_.load()) { stop_reason = "cancelled"; break; }

        // ---- 起草 ----
        llama_tokens draft;
        const llama_token id_last = prompt_tokens.back();

        if (spec) {
            auto & dp = common_speculative_get_draft_params(spec, /*seq_id=*/ 0);
            dp = {
                /* .drafting = */ true,
                /* .n_max    = */ I.cfg.spec_n_max,
                /* .n_past   = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt_tokens,
                /* .result   = */ &draft,
            };
            common_speculative_draft(spec);
            stats_.n_drafted += (int32_t) draft.size();
        }

        // ---- 组装 batch： [id_last] + draft ----
        const size_t n_tok = 1 + draft.size();
        if (n_tok > (size_t) impl->batch_gen.n_tokens) {
            // 草稿数超出预分配容量，退化为逐个 token 解码
            llama_batch_free(impl->batch_gen);
            impl->batch_gen = llama_batch_init((int32_t) n_tok, 0, 1);
        }

        llama_batch & batch = impl->batch_gen;
        batch.n_tokens = (int32_t) n_tok;

        std::vector<int32_t> i_batch;
        i_batch.reserve(n_tok);
        i_batch.push_back(0);
        for (size_t k = 0; k < draft.size(); k++) i_batch.push_back((int32_t) (k + 1));

        llama_pos pos = n_past;
        {
            const size_t k = 0;
            batch.token[k]     = id_last;
            batch.pos[k]       = pos++;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k]    = 1;
        }
        for (size_t k = 0; k < draft.size(); k++) {
            const size_t j = k + 1;
            batch.token[j]     = draft[k];
            batch.pos[j]       = pos++;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }

        if (llama_decode(ctx, batch) != 0) {
            err = "解码失败（上下文已满？）";
            stop_reason = "error";
            ok = false;
            break;
        }

        // ---- 验证并接受 ----
        llama_tokens accepted;
        if (spec && !draft.empty()) {
            accepted = common_sampler_sample_and_accept_n(I.smpl, ctx, i_batch, draft);
            common_speculative_accept(spec, /*seq_id=*/ 0,
                                      (uint16_t) (accepted.size() - 1));
            stats_.n_accepted += (int32_t) (accepted.size() - 1);
        } else {
            llama_token t = common_sampler_sample(I.smpl, ctx, /*idx=*/ 0);
            common_sampler_accept(I.smpl, t, true);
            accepted.push_back(t);
        }

        // ---- 输出 ----
        for (size_t k = 0; k < accepted.size(); k++) {
            const llama_token t = accepted[k];
            if (t == eos) { stop_reason = "eos"; break; }

            std::string piece = common_token_to_piece(ctx, t, /*special=*/ false);
            if (!piece.empty() && cb && !cb(piece)) {
                cancelled_.store(true);
                stop_reason = "cancelled";
                break;
            }
            n_gen++;
            n_past++;
            prompt_tokens.push_back(t);

            if (n_gen >= n_predict_max) { stop_reason = "length"; break; }
        }
    }

    if (stop_reason.empty()) stop_reason = "length";

    stats_.t_gen_ms   = now_ms() - t_gen_0;
    stats_.n_gen_tokens = n_gen;
    if (stats_.t_gen_ms > 0) {
        stats_.gen_tps = n_gen / (stats_.t_gen_ms / 1000.0);
    }
    if (stats_.n_drafted > 0) {
        stats_.accept_rate = (double) stats_.n_accepted / (double) stats_.n_drafted;
    }
    stats_.stop_reason = stop_reason;

    if (chunks) mtmd_input_chunks_free(chunks);

    if (I.smpl) { common_sampler_free(I.smpl); I.smpl = nullptr; }

    LM_LOGI("生成结束: %d tokens, %.2f tok/s, 接受率 %.2f (%d/%d), stop=%s",
            n_gen, stats_.gen_tps, stats_.accept_rate,
            stats_.n_accepted, stats_.n_drafted, stop_reason.c_str());

    return true;
}

// ===========================================================================
// Embedder
// ===========================================================================

struct Embedder::Impl {
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    int32_t         n_embd = 0;
    int32_t         n_batch = 2048;
};

Embedder::Embedder()  : impl(new Impl) {}
Embedder::~Embedder() { unload(); }

bool Embedder::is_loaded() const { return impl && impl->ctx != nullptr; }
int32_t Embedder::n_embd() const { return impl->n_embd; }

bool Embedder::load(const std::string & model_path,
                    int32_t n_ctx, int32_t n_gpu_layers, int32_t n_threads,
                    std::string & err) {
    unload();

    if (model_path.empty()) { err = "embedding 模型路径为空"; return false; }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    impl->model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!impl->model) { err = "加载 embedding 模型失败: " + model_path; return false; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = n_ctx;
    cparams.n_batch   = n_ctx;
    cparams.n_ubatch  = n_ctx;
    cparams.n_threads = n_threads;
    cparams.embeddings = true;   // 输出句向量
    // pooling_type 保持 UNSPECIFIED，让 llama.cpp 按模型元数据决定

    impl->ctx = llama_init_from_model(impl->model, cparams);
    if (!impl->ctx) {
        err = "创建 embedding 上下文失败";
        llama_model_free(impl->model);
        impl->model = nullptr;
        return false;
    }

    impl->n_embd  = llama_model_n_embd(impl->model);
    impl->n_batch = n_ctx;
    return true;
}

void Embedder::unload() {
    if (!impl) return;
    if (impl->ctx)   { llama_free(impl->ctx);   impl->ctx = nullptr; }
    if (impl->model) { llama_model_free(impl->model); impl->model = nullptr; }
    impl->n_embd = 0;
}

bool Embedder::embed(const std::vector<std::string> & texts,
                     std::vector<std::vector<float>> & out,
                     std::string & err) {
    if (!impl->ctx) { err = "embedding 模型未加载"; return false; }

    out.clear();
    out.reserve(texts.size());

    llama_context * ctx = impl->ctx;
    llama_memory_t  mem = llama_get_memory(ctx);

    for (const auto & text : texts) {
        std::vector<llama_token> toks = tokenize_text(impl->model, text,
                                                     /*add_special=*/ true,
                                                     /*parse_special=*/ false);
        if (toks.empty()) {
            // 空文本也要占位，否则索引与入库记录错位
            out.emplace_back();
            continue;
        }

        if ((int32_t) toks.size() > impl->n_batch) {
            toks.resize(impl->n_batch);   // 截断，embedding 模型一般足够长
        }

        if (mem) llama_memory_seq_rm(mem, 0, 0, -1);

        llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
        for (size_t k = 0; k < toks.size(); k++) {
            batch.token[k]     = toks[k];
            batch.pos[k]       = (llama_pos) k;
            batch.n_seq_id[k]  = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k]    = 0;   // embedding 不需要 logits
        }

        if (llama_decode(ctx, batch) != 0) {
            llama_batch_free(batch);
            err = "embedding 解码失败";
            return false;
        }
        llama_batch_free(batch);

        const float * emb = llama_get_embeddings_seq(ctx, /*seq_id=*/ 0);
        if (!emb) {
            emb = llama_get_embeddings(ctx);
        }
        if (!emb) {
            err = "模型未输出句向量（该模型是否为生成式模型？）";
            return false;
        }

        std::vector<float> v(emb, emb + impl->n_embd);

        // L2 归一化：之后检索时点积即余弦相似度，可以省掉一次开方
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (float & x : v) x /= norm;
        }
        out.push_back(std::move(v));
    }
    return true;
}

// ===========================================================================
// 运行时探测
// ===========================================================================

static int64_t read_meminfo_total_kb() {
    std::ifstream f("/proc/meminfo");
    if (!f) return 0;
    std::string key;
    int64_t     val = 0;
    std::string unit;
    while (f >> key) {
        if (key == "MemTotal:") {
            f >> val >> unit;
            return val;
        }
        std::getline(f, unit);
    }
    return 0;
}

RuntimeInfo probe_runtime() {
    RuntimeInfo ri;

    ri.n_cores = (int32_t) std::thread::hardware_concurrency();

    const int64_t kb = read_meminfo_total_kb();
    ri.total_ram_mb = kb / 1024;

    // 设备枚举（OpenCL 设备在 ggml 中登记为 GPU 类型）
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t i = 0; i < n_dev; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (!dev) continue;
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            ri.opencl_available = true;
            ri.gpu_name = ggml_backend_dev_description(dev);
            break;
        }
    }

#if defined(__ANDROID__)
    auto get_prop = [](const char * key) -> std::string {
        char buf[256] = {0};
        __system_property_get(key, buf);
        return std::string(buf);
    };
    ri.soc_model = get_prop("ro.soc.model");
    if (ri.soc_model.empty()) ri.soc_model = get_prop("ro.board.platform");
    if (ri.gpu_name.empty())  ri.gpu_name  = get_prop("ro.hardware.vulkan");
#endif

    return ri;
}

}  // namespace localmind
