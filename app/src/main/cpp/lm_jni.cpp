// JNI 绑定层。
//
// 约定：
//   - 生成是阻塞调用，Kotlin 侧必须在 Dispatchers.IO / Default 上发起。
//   - 配置与统计用 JSON 传递，避免为几十个参数定义大量 JNI 方法。
//   - 图片由 Kotlin 侧解码后以裸 RGB 字节数组传入，C++ 不做图像解码。

#include <jni.h>

#include "lm_engine.h"
#include "lm_log.h"

#include "common.h"

// 注意：llama.cpp 的 common/json.h 是它自己的 pimpl 封装，不是 nlohmann。
// nlohmann 位于 llama.cpp/vendor/nlohmann（common 库自身也依赖它），
// 因此只要 llama.cpp 能编译，这个头文件就一定存在。
#include "nlohmann/json.hpp"

#include <cstring>
#include <string>
#include <vector>

using namespace localmind;
using json = nlohmann::json;

static JavaVM * g_vm = nullptr;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    return JNI_VERSION_1_6;
}

// ---------------------------------------------------------------------------
// 把 llama.cpp 的日志接到 logcat
// ---------------------------------------------------------------------------
static void llama_log_sink(ggml_log_level level, const char * text, void *) {
    // llama.cpp 的日志自带换行，这里去掉尾部空白避免 logcat 噪音
    std::string s(text ? text : "");
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    if (s.empty()) return;

    switch (level) {
        case GGML_LOG_LEVEL_ERROR: LM_LOGE("%s", s.c_str()); break;
        case GGML_LOG_LEVEL_WARN:  LM_LOGW("%s", s.c_str()); break;
        case GGML_LOG_LEVEL_INFO:  LM_LOGI("%s", s.c_str()); break;
        default:                   LM_LOGD("%s", s.c_str()); break;
    }
}

// ---------------------------------------------------------------------------
// 帮助函数
// ---------------------------------------------------------------------------
static std::string jstr(JNIEnv * env, jstring s) {
    if (!s) return {};
    const char * c = env->GetStringUTFChars(s, nullptr);
    if (!c) return {};
    std::string out(c);
    env->ReleaseStringUTFChars(s, c);
    return out;
}

static ModelConfig parse_model_config(const json & j) {
    ModelConfig c;
    auto get_s = [&](const char * k, std::string & dst) {
        if (j.contains(k) && j[k].is_string()) dst = j[k].get<std::string>();
    };
    auto get_i = [&](const char * k, int32_t & dst) {
        if (j.contains(k) && j[k].is_number_integer()) dst = j[k].get<int32_t>();
    };
    auto get_f = [&](const char * k, float & dst) {
        if (j.contains(k) && j[k].is_number_float()) dst = j[k].get<float>();
    };
    auto get_b = [&](const char * k, bool & dst) {
        if (j.contains(k) && j[k].is_boolean()) dst = j[k].get<bool>();
    };

    get_s("modelPath",  c.model_path);
    get_s("mmprojPath", c.mmproj_path);
    get_s("draftPath",  c.draft_path);

    get_i("nCtx",     c.n_ctx);
    get_i("nBatch",   c.n_batch);
    get_i("nUbatch",  c.n_ubatch);
    get_i("nGpuLayers", c.n_gpu_layers);
    get_i("nThreads", c.n_threads);
    get_i("nThreadsBatch", c.n_threads_batch);

    get_b("flashAttn", c.flash_attn);

    if (j.contains("cacheTypeK")) c.cache_type_k = (CacheType) j["cacheTypeK"].get<int>();
    if (j.contains("cacheTypeV")) c.cache_type_v = (CacheType) j["cacheTypeV"].get<int>();
    if (j.contains("specMode"))   c.spec_mode    = (SpecMode)  j["specMode"].get<int>();

    get_i("specNMax", c.spec_n_max);
    get_i("specNMin", c.spec_n_min);
    get_f("specPMin", c.spec_p_min);

    get_b("loadMtp", c.load_mtp);

    return c;
}

static SamplerConfig parse_sampler_config(const json & j) {
    SamplerConfig s;
    if (j.contains("temp"))            s.temp            = j["temp"].get<float>();
    if (j.contains("topP"))            s.top_p           = j["topP"].get<float>();
    if (j.contains("topK"))            s.top_k           = j["topK"].get<float>();
    if (j.contains("minP"))            s.min_p           = j["minP"].get<float>();
    if (j.contains("repeatPenalty"))   s.repeat_penalty  = j["repeatPenalty"].get<float>();
    if (j.contains("repeatLastN"))     s.repeat_last_n   = j["repeatLastN"].get<int32_t>();
    if (j.contains("nPredict"))        s.n_predict       = j["nPredict"].get<int32_t>();
    if (j.contains("seed"))            s.seed            = j["seed"].get<uint32_t>();
    return s;
}

// ---------------------------------------------------------------------------
// Token 回调：从 native 线程回调 Java
// ---------------------------------------------------------------------------
struct SinkCtx {
    JavaVM    * vm  = nullptr;
    jobject     obj = nullptr;   // 全局引用
    jmethodID   mid = nullptr;
    bool        stop = false;
};

static bool invoke_sink(SinkCtx & sink, const std::string & piece) {
    if (!sink.vm || !sink.obj || !sink.mid) return true;

    JNIEnv * env = nullptr;
    if (sink.vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK || env == nullptr) {
#if defined(__ANDROID__)
        if (sink.vm->AttachCurrentThread(&env, nullptr) != JNI_OK || env == nullptr) return true;
#else
        if (sink.vm->AttachCurrentThread((void **) &env, nullptr) != JNI_OK || env == nullptr) return true;
#endif
    }

    jstring js = env->NewStringUTF(piece.c_str());
    if (!js) return true;
    jboolean keep = env->CallBooleanMethod(sink.obj, sink.mid, js);
    env->DeleteLocalRef(js);

    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
        return false;
    }
    return keep == JNI_TRUE;
}

// ---------------------------------------------------------------------------
// NativeEngine
// ---------------------------------------------------------------------------
extern "C" {

JNIEXPORT jlong JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeLoad(JNIEnv * env, jclass, jstring cfgJson) {
    llama_log_set(llama_log_sink, nullptr);

    json j = json::parse(jstr(env, cfgJson), nullptr, false);
    if (j.is_discarded()) return 0;

    ModelConfig cfg = parse_model_config(j);

    auto * engine = new Engine();
    std::string err;
    if (!engine->load(cfg, err)) {
        LM_LOGE("load failed: %s", err.c_str());
        delete engine;
        return 0;
    }
    return (jlong) (intptr_t) engine;
}

JNIEXPORT void JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeUnload(JNIEnv *, jclass, jlong handle) {
    if (!handle) return;
    auto * engine = (Engine *) (intptr_t) handle;
    delete engine;
}

JNIEXPORT void JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeCancel(JNIEnv *, jclass, jlong handle) {
    if (!handle) return;
    ((Engine *) (intptr_t) handle)->cancel();
}

JNIEXPORT jstring JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeGenerate(
        JNIEnv * env, jclass,
        jlong handle,
        jstring prompt,
        jobjectArray images,     // Array<ImagePayload>，可为 null
        jstring paramsJson,
        jobject sink) {

    if (!handle) return env->NewStringUTF("{\"error\":\"引擎未初始化\"}");

    auto * engine = (Engine *) (intptr_t) handle;

    std::string text = jstr(env, prompt);

    json jp = paramsJson ? json::parse(jstr(env, paramsJson), nullptr, false)
                         : json::object();
    if (jp.is_discarded()) jp = json::object();
    SamplerConfig sp = parse_sampler_config(jp);

    // ---- 图片 ----
    std::vector<ImageInput> imgs;
    if (images != nullptr) {
        const jsize n = env->GetArrayLength(images);
        for (jsize i = 0; i < n; i++) {
            jobject payload = env->GetObjectArrayElement(images, i);
            if (!payload) continue;

            jclass cls = env->GetObjectClass(payload);
            jfieldID fid_nx  = env->GetFieldID(cls, "nx",  "I");
            jfieldID fid_ny  = env->GetFieldID(cls, "ny",  "I");
            jfieldID fid_rgb = env->GetFieldID(cls, "rgb", "[B");
            if (!fid_nx || !fid_ny || !fid_rgb) {
                env->DeleteLocalRef(cls);
                env->DeleteLocalRef(payload);
                continue;
            }

            ImageInput img;
            img.nx = (uint32_t) env->GetIntField(payload, fid_nx);
            img.ny = (uint32_t) env->GetIntField(payload, fid_ny);

            auto * arr = (jbyteArray) env->GetObjectField(payload, fid_rgb);
            if (arr) {
                const jsize len = env->GetArrayLength(arr);
                img.rgb.resize((size_t) len);
                env->GetByteArrayRegion(arr, 0, len, (jbyte *) img.rgb.data());
                env->DeleteLocalRef(arr);
            }
            imgs.push_back(std::move(img));

            env->DeleteLocalRef(cls);
            env->DeleteLocalRef(payload);
        }
    }

    // ---- 回调 ----
    SinkCtx sink_ctx;
    sink_ctx.vm = g_vm;
    if (sink != nullptr) {
        jclass cls = env->GetObjectClass(sink);
        sink_ctx.mid = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)Z");
        if (sink_ctx.mid) {
            sink_ctx.obj = env->NewGlobalRef(sink);
        }
        env->DeleteLocalRef(cls);
    }

    std::string err;
    bool ok = engine->generate(
            text, imgs, sp,
            [&](const std::string & piece) -> bool {
                return invoke_sink(sink_ctx, piece);
            },
            err);

    if (sink_ctx.obj && g_vm) {
        JNIEnv * e = nullptr;
        if (g_vm->GetEnv((void **) &e, JNI_VERSION_1_6) == JNI_OK && e) {
            e->DeleteGlobalRef(sink_ctx.obj);
        }
    }

    const GenStats & st = engine->stats();
    json r;
    r["ok"]             = ok;
    if (!err.empty()) r["error"] = err;
    r["stopReason"]     = st.stop_reason;
    r["nPromptTokens"]  = st.n_prompt_tokens;
    r["nGenTokens"]     = st.n_gen_tokens;
    r["promptTps"]      = st.prompt_tps;
    r["genTps"]         = st.gen_tps;
    r["nDrafted"]       = st.n_drafted;
    r["nAccepted"]      = st.n_accepted;
    r["acceptRate"]     = st.accept_rate;

    return env->NewStringUTF(r.dump().c_str());
}

JNIEXPORT jstring JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeModelInfo(JNIEnv * env, jclass, jlong handle) {
    json r;
    if (!handle) {
        r["loaded"] = false;
        return env->NewStringUTF(r.dump().c_str());
    }
    auto * e = (Engine *) (intptr_t) handle;
    r["loaded"]   = e->is_loaded();
    r["nCtx"]     = e->n_ctx();
    r["nEmbd"]    = e->n_embd();
    r["vision"]   = e->supports_vision();
    r["desc"]     = e->model_desc();
    r["backend"]  = e->backend_desc();
    return env->NewStringUTF(r.dump().c_str());
}

JNIEXPORT jstring JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeApplyTemplate(
        JNIEnv * env, jclass, jlong handle, jstring messagesJson) {

    if (!handle) return env->NewStringUTF("");

    json j = json::parse(jstr(env, messagesJson), nullptr, false);
    if (j.is_discarded() || !j.is_array()) return env->NewStringUTF("");

    std::vector<ChatMsg> msgs;
    msgs.reserve(j.size());
    for (const auto & item : j) {
        ChatMsg m;
        if (item.contains("role"))    m.role    = item["role"].get<std::string>();
        if (item.contains("content")) m.content = item["content"].get<std::string>();
        msgs.push_back(std::move(m));
    }

    auto * e = (Engine *) (intptr_t) handle;
    return env->NewStringUTF(e->apply_template(msgs, /*add_generation_prompt=*/ true).c_str());
}

// ---------------------------------------------------------------------------
// Embedder
// ---------------------------------------------------------------------------
JNIEXPORT jlong JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeEmbedderLoad(
        JNIEnv * env, jclass, jstring path, jint nCtx, jint nGpuLayers, jint nThreads) {
    auto * emb = new Embedder();
    std::string err;
    if (!emb->load(jstr(env, path), nCtx, nGpuLayers, nThreads, err)) {
        LM_LOGE("embedder load failed: %s", err.c_str());
        delete emb;
        return 0;
    }
    return (jlong) (intptr_t) emb;
}

JNIEXPORT void JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeEmbedderUnload(JNIEnv *, jclass, jlong handle) {
    if (handle) delete (Embedder *) (intptr_t) handle;
}

JNIEXPORT jobjectArray JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeEmbed(JNIEnv * env, jclass,
                                                      jlong handle, jobjectArray texts) {
    if (!handle || !texts) return nullptr;

    auto * emb = (Embedder *) (intptr_t) handle;

    const jsize n = env->GetArrayLength(texts);
    std::vector<std::string> inputs;
    inputs.reserve(n);
    for (jsize i = 0; i < n; i++) {
        auto s = (jstring) env->GetObjectArrayElement(texts, i);
        inputs.push_back(jstr(env, s));
        if (s) env->DeleteLocalRef(s);
    }

    std::vector<std::vector<float>> vecs;
    std::string err;
    if (!emb->embed(inputs, vecs, err)) {
        LM_LOGE("embed failed: %s", err.c_str());
        return nullptr;
    }

    jclass floatCls = env->FindClass("[F");
    if (!floatCls) return nullptr;
    jobjectArray out = env->NewObjectArray((jsize) vecs.size(), floatCls, nullptr);
    for (size_t i = 0; i < vecs.size(); i++) {
        jfloatArray arr = env->NewFloatArray((jsize) vecs[i].size());
        env->SetFloatArrayRegion(arr, 0, (jsize) vecs[i].size(), vecs[i].data());
        env->SetObjectArrayElement(out, (jsize) i, arr);
        env->DeleteLocalRef(arr);
    }
    env->DeleteLocalRef(floatCls);
    return out;
}

JNIEXPORT jint JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeEmbedDim(JNIEnv *, jclass, jlong handle) {
    return handle ? ((Embedder *) (intptr_t) handle)->n_embd() : 0;
}

// ---------------------------------------------------------------------------
// 运行时探测
// ---------------------------------------------------------------------------
JNIEXPORT jstring JNICALL
Java_com_localmind_ai_engine_NativeEngine_nativeProbe(JNIEnv * env, jclass) {
    RuntimeInfo ri = probe_runtime();
    json r;
    r["socModel"]       = ri.soc_model;
    r["gpuName"]        = ri.gpu_name;
    r["nCores"]         = ri.n_cores;
    r["totalRamMb"]     = ri.total_ram_mb;
    r["openclAvailable"] = ri.opencl_available;
    return env->NewStringUTF(r.dump().c_str());
}

}  // extern "C"
