package com.localmind.ai.engine

import com.localmind.ai.api.ApiModel
import com.localmind.ai.api.InferenceBackend
import com.localmind.ai.rag.Embedder
import com.localmind.ai.rag.HybridRetriever

// 聚合所有本地引擎，对外提供统一的推理后端。
//
// 这一层是 HTTP 服务（OpenAiRoutes）与 AIDL 服务共用的"事实来源"：
// 不论是别的 App 通过绑定调用，还是本 App 的 UI，都走同一份模型、同一份知识库。
// 模型本身只加载一份，避免多份权重挤爆内存。
class LocalMindBackend(
    private val runtime: ModelRuntime,
    private val embedder: Embedder,
    private val retriever: HybridRetriever,
    private val modelIdProvider: () -> String,
) : InferenceBackend {

    override val isReady: Boolean get() = runtime.isLoaded
    override val modelId: String get() = modelIdProvider()

    override fun models(): List<ApiModel> {
        val id = modelIdProvider()
        return if (id.isEmpty()) emptyList() else listOf(ApiModel(id))
    }

    override suspend fun renderChat(messages: List<Pair<String, String>>): String? =
        runtime.renderTemplate(messages)

    override suspend fun generateRaw(
        prompt: String,
        images: List<ImagePayload>?,
        sampler: SamplerConfig,
        onToken: ((String) -> Boolean)?,
    ): GenStats = runtime.generate(prompt, images, sampler, onToken)

    override suspend fun embed(texts: List<String>): List<FloatArray>? =
        embedder.embed(texts)

    override suspend fun ragSearch(query: String, topK: Int): List<Pair<String, String>> =
        retriever.retrieve(query, topK).map { it.docTitle to it.content }
}
