package com.localmind.ai.engine

import org.json.JSONObject

// 传给 native 的原始 RGB 位图。
// 用 @JvmField 暴露为 public 字段，供 JNI 直接 GetFieldID 读取，避免反射调用开销。
class ImagePayload(
    @JvmField val nx: Int,
    @JvmField val ny: Int,
    @JvmField val rgb: ByteArray,
)

// 逐 token 回调。返回 false 中止生成。
// native 线程会回调到这里，实现里不要做重活。
interface TokenSink {
    fun onToken(text: String): Boolean
}

enum class SpecMode(val id: Int) {
    OFF(0),
    MTP(1),          // 模型自带 MTP 头，移动端首选
    DRAFT_MODEL(2),  // 独立小 draft 模型
    NGRAM(3),        // ngram 自投机，零额外内存
}

enum class CacheType(val id: Int) {
    F16(0),
    Q8_0(1),
    Q4_0(2),
}

data class ModelConfig(
    val modelPath: String,
    val mmprojPath: String = "",
    val draftPath: String = "",
    val nCtx: Int = 8192,
    val nBatch: Int = 2048,
    val nUbatch: Int = 512,
    val nGpuLayers: Int = -1,
    val nThreads: Int = 4,
    val nThreadsBatch: Int = 0,
    val flashAttn: Boolean = true,
    val cacheTypeK: CacheType = CacheType.F16,
    val cacheTypeV: CacheType = CacheType.F16,
    val specMode: SpecMode = SpecMode.MTP,
    val specNMax: Int = 16,
    val specNMin: Int = 4,
    val specPMin: Float = 0.75f,
    val loadMtp: Boolean = true,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("modelPath", modelPath)
        put("mmprojPath", mmprojPath)
        put("draftPath", draftPath)
        put("nCtx", nCtx)
        put("nBatch", nBatch)
        put("nUbatch", nUbatch)
        put("nGpuLayers", nGpuLayers)
        put("nThreads", nThreads)
        put("nThreadsBatch", nThreadsBatch)
        put("flashAttn", flashAttn)
        put("cacheTypeK", cacheTypeK.id)
        put("cacheTypeV", cacheTypeV.id)
        put("specMode", specMode.id)
        put("specNMax", specNMax)
        put("specNMin", specNMin)
        put("specPMin", specPMin.toDouble())
        put("loadMtp", loadMtp)
    }
}

data class SamplerConfig(
    val temp: Float = 0.7f,
    val topP: Float = 0.9f,
    val topK: Float = 40f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    val nPredict: Int = -1,
    val seed: Int = 0,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("temp", temp.toDouble())
        put("topP", topP.toDouble())
        put("topK", topK.toDouble())
        put("minP", minP.toDouble())
        put("repeatPenalty", repeatPenalty.toDouble())
        put("repeatLastN", repeatLastN)
        put("nPredict", nPredict)
        put("seed", seed)
    }
}

data class GenStats(
    val ok: Boolean,
    val error: String?,
    val stopReason: String,
    val nPromptTokens: Int,
    val nGenTokens: Int,
    val promptTps: Double,
    val genTps: Double,
    val nDrafted: Int,
    val nAccepted: Int,
    val acceptRate: Double,
) {
    companion object {
        fun fromJson(j: JSONObject): GenStats = GenStats(
            ok = j.optBoolean("ok", false),
            error = j.optString("error").takeIf { it.isNotEmpty() },
            stopReason = j.optString("stopReason", ""),
            nPromptTokens = j.optInt("nPromptTokens", 0),
            nGenTokens = j.optInt("nGenTokens", 0),
            promptTps = j.optDouble("promptTps", 0.0),
            genTps = j.optDouble("genTps", 0.0),
            nDrafted = j.optInt("nDrafted", 0),
            nAccepted = j.optInt("nAccepted", 0),
            acceptRate = j.optDouble("acceptRate", 0.0),
        )
    }
}

data class ModelInfo(
    val loaded: Boolean,
    val nCtx: Int,
    val nEmbd: Int,
    val vision: Boolean,
    val desc: String,
    val backend: String,
) {
    companion object {
        fun fromJson(j: JSONObject): ModelInfo = ModelInfo(
            loaded = j.optBoolean("loaded", false),
            nCtx = j.optInt("nCtx", 0),
            nEmbd = j.optInt("nEmbd", 0),
            vision = j.optBoolean("vision", false),
            desc = j.optString("desc", ""),
            backend = j.optString("backend", ""),
        )
    }
}

data class RuntimeInfo(
    val socModel: String,
    val gpuName: String,
    val nCores: Int,
    val totalRamMb: Long,
    val openclAvailable: Boolean,
) {
    companion object {
        fun fromJson(j: JSONObject): RuntimeInfo = RuntimeInfo(
            socModel = j.optString("socModel", ""),
            gpuName = j.optString("gpuName", ""),
            nCores = j.optInt("nCores", 0),
            totalRamMb = j.optLong("totalRamMb", 0L),
            openclAvailable = j.optBoolean("openclAvailable", false),
        )
    }
}

// native 方法的直接映射。
//
// 重要：nativeGenerate 是阻塞调用，会占用调用线程直到生成结束。
// 必须在 Dispatchers.IO / Default 上发起，绝不能在主线程调用。
object NativeEngine {

    private var loaded = false

    fun ensureLoaded() {
        if (loaded) return
        System.loadLibrary("localmind")
        loaded = true
    }

    external fun nativeLoad(cfgJson: String): Long
    external fun nativeUnload(handle: Long)
    external fun nativeCancel(handle: Long)

    external fun nativeGenerate(
        handle: Long,
        prompt: String,
        images: Array<ImagePayload>?,
        paramsJson: String,
        sink: TokenSink?,
    ): String

    external fun nativeModelInfo(handle: Long): String

    // 按模型自带模板渲染对话。返回空串表示模型未提供模板。
    external fun nativeApplyTemplate(handle: Long, messagesJson: String): String

    external fun nativeEmbedderLoad(
        path: String, nCtx: Int, nGpuLayers: Int, nThreads: Int,
    ): Long

    external fun nativeEmbedderUnload(handle: Long)
    external fun nativeEmbed(handle: Long, texts: Array<String>): Array<FloatArray>?
    external fun nativeEmbedDim(handle: Long): Int

    external fun nativeProbe(): String

    fun probe(): RuntimeInfo = runCatching {
        RuntimeInfo.fromJson(JSONObject(nativeProbe()))
    }.getOrDefault(RuntimeInfo("", "", 0, 0L, false))

    fun modelInfo(handle: Long): ModelInfo = runCatching {
        ModelInfo.fromJson(JSONObject(nativeModelInfo(handle)))
    }.getOrDefault(ModelInfo(false, 0, 0, false, "", ""))

    fun generate(
        handle: Long,
        prompt: String,
        images: List<ImagePayload>?,
        sampler: SamplerConfig,
        sink: TokenSink?,
    ): GenStats {
        val arr = images?.takeIf { it.isNotEmpty() }?.toTypedArray()
        val raw = nativeGenerate(handle, prompt, arr, sampler.toJson().toString(), sink)
        return GenStats.fromJson(JSONObject(raw))
    }

    // messages: [{"role":"user","content":"..."}, ...]
    // 返回渲染后的完整 prompt；若模型无模板则返回 null，
    // 调用方应回退到简单的 role 前缀拼接。
    fun applyTemplate(handle: Long, messages: List<Pair<String, String>>): String? {
        val arr = org.json.JSONArray()
        messages.forEach { (role, content) ->
            arr.put(JSONObject().apply {
                put("role", role)
                put("content", content)
            })
        }
        return runCatching { nativeApplyTemplate(handle, arr.toString()) }
            .getOrNull()?.takeIf { it.isNotEmpty() }
    }
}
