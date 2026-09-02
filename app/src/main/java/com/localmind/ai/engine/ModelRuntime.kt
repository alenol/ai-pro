package com.localmind.ai.engine

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

// 单个主模型的生命周期管理。
//
// 关于并发：llama.cpp 的一个 llama_context 不支持并发生成，
// 两个线程同时调 llama_decode 会直接破坏 KV cache。
// 因此这里用 Mutex 把生成串行化 —— 来自 UI 的请求和来自本地 API 的请求
// 会排队执行，而不是互相踩踏。
//
// 这也解释了为什么本地 API 只能串行服务：想要并发就得开多个 slot
// （每个 slot 一份 KV cache），在手机上那是拿内存换并发，不划算。

class ModelRuntime {

    @Volatile private var handle: Long = 0L
    @Volatile private var currentConfig: ModelConfig? = null

    private val genLock = Mutex()

    val isLoaded: Boolean get() = handle != 0L
    val config: ModelConfig? get() = currentConfig

    fun init() { NativeEngine.ensureLoaded() }

    suspend fun load(cfg: ModelConfig): Result<ModelInfo> = withContext(Dispatchers.IO) {
        genLock.withLock {
            unloadInternal()
            init()
            val h = NativeEngine.nativeLoad(cfg.toJson().toString())
            if (h == 0L) {
                Result.failure(IllegalStateException("模型加载失败，请检查路径与内存是否充足"))
            } else {
                handle = h
                currentConfig = cfg
                Result.success(NativeEngine.modelInfo(h))
            }
        }
    }

    suspend fun unload() = withContext(Dispatchers.IO) {
        genLock.withLock { unloadInternal() }
    }

    private fun unloadInternal() {
        if (handle != 0L) {
            NativeEngine.nativeUnload(handle)
            handle = 0L
            currentConfig = null
        }
    }

    fun info(): ModelInfo =
        if (handle == 0L) ModelInfo(false, 0, 0, false, "", "")
        else NativeEngine.modelInfo(handle)

    fun cancel() {
        if (handle != 0L) NativeEngine.nativeCancel(handle)
    }

    // 阻塞式生成。onToken 在 native 线程被调用，实现里不要做重活。
    suspend fun generate(
        prompt: String,
        images: List<ImagePayload>? = null,
        sampler: SamplerConfig = SamplerConfig(),
        onToken: ((String) -> Boolean)? = null,
    ): GenStats = withContext(Dispatchers.IO) {
        genLock.withLock {
            if (handle == 0L) return@withLock GenStats(
                ok = false, error = "模型未加载", stopReason = "error",
                nPromptTokens = 0, nGenTokens = 0, promptTps = 0.0, genTps = 0.0,
                nDrafted = 0, nAccepted = 0, acceptRate = 0.0,
            )
            NativeEngine.generate(handle, prompt, images, sampler, onToken?.let { sink ->
                object : TokenSink {
                    override fun onToken(text: String): Boolean = sink(text)
                }
            })
        }
    }

    fun renderTemplate(messages: List<Pair<String, String>>): String? =
        if (handle == 0L) null else NativeEngine.applyTemplate(handle, messages)
}
