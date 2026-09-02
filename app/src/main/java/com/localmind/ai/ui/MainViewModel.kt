package com.localmind.ai.ui

import android.app.Application
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.localmind.ai.LocalMindApplication
import com.localmind.ai.engine.GenStats
import com.localmind.ai.engine.ImagePayload
import com.localmind.ai.engine.ModelInfo
import com.localmind.ai.engine.PerfPreset
import com.localmind.ai.engine.SamplerConfig
import com.localmind.ai.engine.SnapdragonTuner
import com.localmind.ai.engine.SpecMode
import com.localmind.ai.rag.DocRow
import com.localmind.ai.rag.DocumentIngestor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class UiMessage(
    val id: Long,
    val role: String,          // "user" | "assistant" | "system"
    val text: String,
    val images: List<ImagePayload> = emptyList(),
    val isStreaming: Boolean = false,
    val error: String? = null,
)

class MainViewModel(app: Application) : AndroidViewModel(app) {

    private val svc = (app as LocalMindApplication).services
    private val ingestor = DocumentIngestor(app, svc.db, svc.embedder).apply {
        // 多模态 OCR：PDF / 图片导入时，把页面渲染图交给视觉模型识别成文本
        visionExtractor = { payload ->
            val sb = StringBuilder()
            val stats = svc.runtime.generate(
                prompt = "把这张图里的文字完整提取出来，只输出文字内容。",
                images = listOf(payload),
                sampler = SamplerConfig(nPredict = 1024),
            ) { sb.append(it); true }
            if (stats.error != null) error(stats.error!!)
            sb.toString()
        }
    }

    val profile = svc.profile

    // -----------------------------------------------------------------
    // 模型
    // -----------------------------------------------------------------
    var modelLoaded by mutableStateOf(false)
        private set
    var modelInfo by mutableStateOf<ModelInfo?>(null)
        private set
    var modelPath by mutableStateOf("")
    var mmprojPath by mutableStateOf("")
    var draftPath by mutableStateOf("")
    var preset by mutableStateOf(PerfPreset.BALANCED)

    var busy by mutableStateOf(false)
        private set
    var status by mutableStateOf("未加载模型")

    // -----------------------------------------------------------------
    // 对话
    // -----------------------------------------------------------------
    val messages = mutableStateListOf<UiMessage>()
    var useRag by mutableStateOf(true)
    var generating by mutableStateOf(false)
        private set
    var lastStats by mutableStateOf<GenStats?>(null)
        private set
    var genHint by mutableStateOf("")
        private set

    val hasKnowledge: Boolean get() = svc.db.chunkCount() > 0

    // -----------------------------------------------------------------
    // 知识库
    // -----------------------------------------------------------------
    val docs = mutableStateListOf<DocRow>()
    var ragBusy by mutableStateOf(false)
        private set
    var ragStatus by mutableStateOf("")
        private set

    // -----------------------------------------------------------------
    // 本地 API
    // -----------------------------------------------------------------
    var apiEnabled by mutableStateOf(false)
        private set
    var apiEndpoint by mutableStateOf("")
        private set
    var apiPort by mutableStateOf("8080")
    var apiKey by mutableStateOf("")
    var apiLan by mutableStateOf(false)   // 是否监听 0.0.0.0（局域网）

    init {
        refreshDocs()
        apiEndpoint = svc.httpEndpoint()
        apiEnabled = svc.isHttpRunning()
    }

    // -----------------------------------------------------------------
    // 模型加载 / 卸载
    // -----------------------------------------------------------------
    fun loadModel() {
        if (modelPath.isBlank()) { status = "请先选择模型文件"; return }
        viewModelScope.launch {
            busy = true
            status = "加载中…"
            val bytes = runCatching { java.io.File(modelPath).length() }.getOrDefault(0L)
            val res = svc.loadModel(modelPath, mmprojPath, preset, bytes, draftPath)
            busy = false
            res.onSuccess {
                modelLoaded = true
                modelInfo = it
                status = "已加载：${it.desc}  [${it.backend}]"
            }.onFailure { status = "加载失败：${it.message}" }
        }
    }

    fun unload() {
        viewModelScope.launch {
            svc.unloadModel()
            modelLoaded = false
            modelInfo = null
            status = "已卸载"
        }
    }

    // -----------------------------------------------------------------
    // 对话
    // -----------------------------------------------------------------
    fun send(text: String, images: List<ImagePayload> = emptyList()) {
        val content = text.trim()
        if (content.isBlank() && images.isEmpty()) return
        if (generating) return

        val userMsg = UiMessage(
            id = System.nanoTime(), role = "user", text = content, images = images,
        )
        messages.add(userMsg)

        val assistantId = System.nanoTime() + 1
        messages.add(UiMessage(id = assistantId, role = "assistant", text = "", isStreaming = true))

        generating = true
        genHint = ""

        viewModelScope.launch {
            val sampler = SamplerConfig()
            val conversation = messages
                .filter { it.id != assistantId }   // 排除当前流式占位
                .map { it.role to it.text }
                .toMutableList()

            // 知识库检索并注入上下文（作为 system 段插到最前）
            if (useRag && hasKnowledge) {
                val hits = svc.retriever.retrieve(content, 5)
                if (hits.isNotEmpty()) {
                    val ctx = svc.retriever.buildContext(hits)
                    conversation.add(0, "system" to "请基于以下本地资料回答，若资料未提及可结合常识：\n$ctx")
                }
            }

            val rendered = withContext(Dispatchers.IO) { svc.runtime.renderTemplate(conversation) }
            val prompt = rendered ?: simpleConcat(conversation)

            val sb = StringBuilder()
            val stats = svc.runtime.generate(prompt, images.ifEmpty { null }, sampler) {
                sb.append(it)
                val msg = messages.firstOrNull { m -> m.id == assistantId }
                if (msg != null) {
                    val idx = messages.indexOf(msg)
                    messages[idx] = msg.copy(text = sb.toString())
                }
                true
            }

            // 收尾：写入完整文本并关闭 streaming 标记
            val idx = messages.indexOfFirst { it.id == assistantId }
            if (idx >= 0) {
                messages[idx] = messages[idx].copy(
                    text = sb.toString(), isStreaming = false,
                    error = stats.error,
                )
            }
            lastStats = stats
            genHint = if (stats.error != null) "错误：${stats.error}"
            else "速度 ${"%.1f".format(stats.genTps)} tok/s" +
                 if (stats.nDrafted > 0) " · 接受率 ${"%.0f".format(stats.acceptRate * 100)}%" else ""

            generating = false
        }
    }

    fun cancel() {
        svc.runtime.cancel()
    }

    private fun simpleConcat(msgs: List<Pair<String, String>>): String =
        msgs.joinToString("\n") { (r, c) -> "${r.uppercase()}: $c" } + "\nASSISTANT:"

    // -----------------------------------------------------------------
    // 知识库
    // -----------------------------------------------------------------
    fun refreshDocs() {
        docs.clear()
        docs.addAll(svc.db.listDocuments())
    }

    fun importDocument(uri: Uri) {
        if (svc.embedder.isReady.not()) {
            ragStatus = "请先加载 embedding 模型（设置页），否则只能建纯 BM25 索引"
        }
        viewModelScope.launch {
            ragBusy = true
            ragStatus = "导入中…"
            val res = ingestor.ingestUri(uri) { ragStatus = it }
            ragBusy = false
            res.onSuccess {
                ragStatus = "完成：${it.nChunks} 段"
                refreshDocs()
            }.onFailure { ragStatus = "导入失败：${it.message}" }
        }
    }

    fun deleteDoc(id: Long) {
        svc.db.deleteDocument(id)
        svc.db.invalidateVectorCache()
        refreshDocs()
    }

    fun clearKnowledge() {
        svc.db.clearAll()
        svc.db.invalidateVectorCache()
        refreshDocs()
        ragStatus = "知识库已清空"
    }

    // -----------------------------------------------------------------
    // 本地 API
    // -----------------------------------------------------------------
    fun toggleApi() {
        if (apiEnabled) {
            svc.stopHttp()
            apiEnabled = false
            apiEndpoint = ""
        } else {
            val port = apiPort.toIntOrNull() ?: 8080
            val host = if (apiLan) "0.0.0.0" else "127.0.0.1"
            val ok = svc.startHttp(port, host, apiKey.trim())
            apiEnabled = ok
            apiEndpoint = if (ok) svc.httpEndpoint() else ""
            if (!ok) ragStatus = "HTTP 服务启动失败，端口可能被占用"
        }
    }

    // 给外部分享用的调优解释文案
    fun tuningTips(): List<String> {
        val bytes = runCatching { java.io.File(modelPath).length() }.getOrDefault(0L)
        val cfg = if (modelLoaded) svc.runtime.config else null
        return if (cfg != null) SnapdragonTuner.explain(profile, cfg, bytes / (1024 * 1024))
        else listOf("模型未加载，无法生成建议")
    }
}
