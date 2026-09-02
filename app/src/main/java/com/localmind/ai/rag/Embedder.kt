package com.localmind.ai.rag

import com.localmind.ai.engine.NativeEngine

// 向量化器。
//
// 与主生成引擎分离：embedding 模型通常只有 0.3B~0.6B，独立加载/卸载
// 意味着用户没开知识库时完全不占内存。
//
// 所有方法都是阻塞的（native 调用），请在 Dispatchers.IO 上调用。

class Embedder {

    @Volatile private var handle: Long = 0L
    private val lock = Any()

    val isReady: Boolean get() = handle != 0L

    fun load(
        path: String,
        nCtx: Int = 2048,
        nGpuLayers: Int = -1,
        nThreads: Int = 3,
    ): Boolean = synchronized(lock) {
        if (handle != 0L) return true
        NativeEngine.ensureLoaded()
        val h = NativeEngine.nativeEmbedderLoad(path, nCtx, nGpuLayers, nThreads)
        handle = h
        h != 0L
    }

    fun unload() = synchronized(lock) {
        if (handle != 0L) {
            NativeEngine.nativeEmbedderUnload(handle)
            handle = 0L
        }
    }

    fun dim(): Int = if (handle == 0L) 0 else NativeEngine.nativeEmbedDim(handle)

    // 返回与输入等长的结果；失败时返回 null。
    // 空文本会返回空向量，调用方需要注意保持索引对齐。
    fun embed(texts: List<String>): List<FloatArray>? {
        if (handle == 0L || texts.isEmpty()) return null
        return synchronized(lock) {
            NativeEngine.nativeEmbed(handle, texts.toTypedArray())?.toList()
        }
    }

    fun embedOne(text: String): FloatArray? = embed(listOf(text))?.firstOrNull()
}
