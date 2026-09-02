package com.localmind.ai.api

import com.localmind.ai.engine.GenStats
import com.localmind.ai.engine.ImagePayload
import com.localmind.ai.engine.SamplerConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

// OpenAI 兼容的 HTTP 路由。
//
// 兼容的目标不是"实现完整的 OpenAI API"，而是让现有的
// OpenAI SDK / 各类第三方客户端（如支持自定义端点的 App）能直接连上来：
// 只要 /v1/chat/completions、/v1/models、/v1/embeddings 三个端点对得上，
// 绝大多数客户端就能开箱使用。

data class ApiModel(val id: String, val ownedBy: String = "localmind")

interface InferenceBackend {
    val isReady: Boolean
    val modelId: String

    fun models(): List<ApiModel>

    /** 渲染对话模板；模型无模板时返回 null */
    suspend fun renderChat(messages: List<Pair<String, String>>): String?

    suspend fun generateRaw(
        prompt: String,
        images: List<ImagePayload>?,
        sampler: SamplerConfig,
        onToken: ((String) -> Boolean)?,
    ): GenStats

    suspend fun embed(texts: List<String>): List<FloatArray>?

    /** 知识库检索，返回 (标题, 内容) */
    suspend fun ragSearch(query: String, topK: Int): List<Pair<String, String>>
}

class OpenAiRoutes(private val backend: InferenceBackend) : HttpHandler {

    override suspend fun handle(req: HttpRequest, out: ResponseWriter): Boolean =
        when {
            req.path == "/health" || req.path == "/" -> { jsonOk(out, healthJson()); true }
            req.path == "/v1/models" -> { jsonOk(out, modelsJson()); true }
            req.path == "/v1/chat/completions" && req.method == "POST" -> {
                chatCompletions(req, out); true
            }
            req.path == "/v1/completions" && req.method == "POST" -> { completions(req, out); true }
            req.path == "/v1/embeddings" && req.method == "POST" -> { embeddings(req, out); true }
            req.path == "/v1/rag/search" && req.method == "POST" -> { ragSearch(req, out); true }
            else -> false
        }

    // -----------------------------------------------------------------
    private fun healthJson(): JSONObject = JSONObject().apply {
        put("status", if (backend.isReady) "ok" else "model_not_loaded")
        put("model", backend.modelId)
        put("ready", backend.isReady)
    }

    private fun modelsJson(): JSONObject {
        val arr = JSONArray()
        backend.models().forEach { m ->
            arr.put(JSONObject().apply {
                put("id", m.id)
                put("object", "model")
                put("created", 0)
                put("owned_by", m.ownedBy)
            })
        }
        return JSONObject().apply {
            put("object", "list")
            put("data", arr)
        }
    }

    private suspend fun chatCompletions(req: HttpRequest, out: ResponseWriter) {
        val body = runCatching { JSONObject(req.bodyText()) }.getOrNull()
        if (body == null) { error(out, 400, "请求体不是合法 JSON"); return }

        if (!backend.isReady) { error(out, 503, "模型未加载"); return }

        val messages = mutableListOf<Pair<String, String>>()
        body.optJSONArray("messages")?.let { arr ->
            for (i in 0 until arr.length()) {
                val m = arr.optJSONObject(i) ?: continue
                val role = m.optString("role", "user")
                // content 可能是字符串，也可能是 [{type:text,text:...}]
                val content = m.opt("content")
                val text = when (content) {
                    is String -> content
                    is JSONArray -> buildString {
                        for (k in 0 until content.length()) {
                            val part = content.optJSONObject(k) ?: continue
                            if (part.optString("type") == "text") append(part.optString("text", ""))
                        }
                    }
                    else -> ""
                }
                messages += role to text
            }
        }
        if (messages.isEmpty()) { error(out, 400, "messages 为空"); return }

        val stream = body.optBoolean("stream", false)
        val sampler = SamplerConfig(
            temp = body.optDouble("temperature", 0.7).toFloat(),
            topP = body.optDouble("top_p", 0.9).toFloat(),
            nPredict = if (body.has("max_tokens")) body.optInt("max_tokens", 512) else -1,
        )

        // 优先用模型自带模板；拿不到模板时回退到简单拼接
        val prompt = backend.renderChat(messages)
            ?: messages.joinToString("\n") { (role, content) ->
                "${role.uppercase()}: $content"
            } + "\nASSISTANT:"

        val id = "chatcmpl-${UUID.randomUUID()}"
        val created = System.currentTimeMillis() / 1000

        if (!stream) {
            val sb = StringBuilder()
            val stats = withContext(Dispatchers.IO) {
                backend.generateRaw(prompt, null, sampler) { sb.append(it); true }
            }
            if (stats.error != null) { error(out, 500, stats.error!!); return }

            val resp = JSONObject().apply {
                put("id", id)
                put("object", "chat.completion")
                put("created", created)
                put("model", backend.modelId)
                put("choices", JSONArray().put(JSONObject().apply {
                    put("index", 0)
                    put("message", JSONObject().apply {
                        put("role", "assistant")
                        put("content", sb.toString())
                    })
                    put("finish_reason", finishReason(stats.stopReason))
                }))
                put("usage", usageJson(stats))
            }
            jsonOk(out, resp)
        } else {
            out.status(200, "OK")
            out.header("Content-Type", "text/event-stream")
            out.header("Cache-Control", "no-cache")
            out.endHeaders()

            withContext(Dispatchers.IO) {
                backend.generateRaw(prompt, null, sampler) { piece ->
                    val chunk = JSONObject().apply {
                        put("id", id)
                        put("object", "chat.completion.chunk")
                        put("created", created)
                        put("model", backend.modelId)
                        put("choices", JSONArray().put(JSONObject().apply {
                            put("index", 0)
                            put("delta", JSONObject().put("content", piece))
                            put("finish_reason", JSONObject.NULL)
                        }))
                    }
                    out.writeText("data: ${chunk}\n\n")
                    out.flush()
                    true
                }
            }
            val last = JSONObject().apply {
                put("id", id)
                put("object", "chat.completion.chunk")
                put("created", created)
                put("model", backend.modelId)
                put("choices", JSONArray().put(JSONObject().apply {
                    put("index", 0)
                    put("delta", JSONObject())
                    put("finish_reason", "stop")
                }))
            }
            out.writeText("data: $last\n\n")
            out.writeText("data: [DONE]\n\n")
        }
    }

    private suspend fun completions(req: HttpRequest, out: ResponseWriter) {
        val body = runCatching { JSONObject(req.bodyText()) }.getOrNull()
            ?: run { error(out, 400, "请求体不是合法 JSON"); return }
        if (!backend.isReady) { error(out, 503, "模型未加载"); return }

        val prompt = body.optString("prompt", "")
        if (prompt.isEmpty()) { error(out, 400, "prompt 为空"); return }

        val sampler = SamplerConfig(
            temp = body.optDouble("temperature", 0.7).toFloat(),
            topP = body.optDouble("top_p", 0.9).toFloat(),
            nPredict = if (body.has("max_tokens")) body.optInt("max_tokens", 256) else -1,
        )

        val sb = StringBuilder()
        val stats = withContext(Dispatchers.IO) {
            backend.generateRaw(prompt, null, sampler) { sb.append(it); true }
        }
        if (stats.error != null) { error(out, 500, stats.error!!); return }

        jsonOk(out, JSONObject().apply {
            put("id", "cmpl-${UUID.randomUUID()}")
            put("object", "text_completion")
            put("created", System.currentTimeMillis() / 1000)
            put("model", backend.modelId)
            put("choices", JSONArray().put(JSONObject().apply {
                put("index", 0)
                put("text", sb.toString())
                put("finish_reason", finishReason(stats.stopReason))
            }))
            put("usage", usageJson(stats))
        })
    }

    private suspend fun embeddings(req: HttpRequest, out: ResponseWriter) {
        val body = runCatching { JSONObject(req.bodyText()) }.getOrNull()
            ?: run { error(out, 400, "请求体不是合法 JSON"); return }

        val rawInput = body.opt("input")
        val inputs = when (rawInput) {
            is String -> listOf(rawInput)
            is JSONArray -> (0 until rawInput.length()).mapNotNull { rawInput.optString(it) }
            else -> emptyList()
        }
        if (inputs.isEmpty()) { error(out, 400, "input 为空"); return }

        val vecs = withContext(Dispatchers.IO) { backend.embed(inputs) }
        if (vecs == null) { error(out, 503, "embedding 模型未加载"); return }

        val arr = JSONArray()
        vecs.forEachIndexed { i, v ->
            arr.put(JSONObject().apply {
                put("object", "embedding")
                put("index", i)
                put("embedding", JSONArray(v.toList()))
            })
        }
        jsonOk(out, JSONObject().apply {
            put("object", "list")
            put("model", backend.modelId)
            put("data", arr)
        })
    }

    // 非 OpenAI 标准的扩展端点：直接查本地知识库
    private suspend fun ragSearch(req: HttpRequest, out: ResponseWriter) {
        val body = runCatching { JSONObject(req.bodyText()) }.getOrNull()
            ?: run { error(out, 400, "请求体不是合法 JSON"); return }
        val query = body.optString("query", "")
        if (query.isBlank()) { error(out, 400, "query 为空"); return }
        val topK = body.optInt("top_k", 5)

        val hits = withContext(Dispatchers.IO) { backend.ragSearch(query, topK) }
        val arr = JSONArray()
        hits.forEach { (title, content) ->
            arr.put(JSONObject().apply {
                put("title", title)
                put("content", content)
            })
        }
        jsonOk(out, JSONObject().apply { put("results", arr) })
    }

    // -----------------------------------------------------------------
    private fun finishReason(stop: String): String = when (stop) {
        "eos" -> "stop"
        "length" -> "length"
        "cancelled" -> "stop"
        else -> "stop"
    }

    private fun usageJson(s: GenStats): JSONObject = JSONObject().apply {
        put("prompt_tokens", s.nPromptTokens)
        put("completion_tokens", s.nGenTokens)
        put("total_tokens", s.nPromptTokens + s.nGenTokens)
        // 扩展字段：本地推理才有意义的性能数据
        put("prompt_tps", s.promptTps)
        put("gen_tps", s.genTps)
        put("draft_accept_rate", s.acceptRate)
    }

    private fun jsonOk(out: ResponseWriter, obj: JSONObject) {
        val bytes = obj.toString().toByteArray(Charsets.UTF_8)
        out.status(200, "OK")
        out.header("Content-Type", "application/json; charset=utf-8")
        out.header("Content-Length", bytes.size.toString())
        out.endHeaders()
        out.write(bytes)
    }

    private fun error(out: ResponseWriter, code: Int, msg: String) {
        val obj = JSONObject().apply {
            put("error", JSONObject().apply {
                put("message", msg)
                put("type", "localmind_error")
                put("code", code)
            })
        }
        out.status(code, if (code == 503) "Service Unavailable" else "Bad Request")
        jsonOkBody(out, obj)
    }

    private fun jsonOkBody(out: ResponseWriter, obj: JSONObject) {
        val bytes = obj.toString().toByteArray(Charsets.UTF_8)
        out.header("Content-Type", "application/json; charset=utf-8")
        out.header("Content-Length", bytes.size.toString())
        out.endHeaders()
        out.write(bytes)
    }
}
