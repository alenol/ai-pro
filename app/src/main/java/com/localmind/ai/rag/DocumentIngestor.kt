package com.localmind.ai.rag

import android.content.Context
import android.graphics.Bitmap
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.ParcelFileDescriptor
import com.localmind.ai.engine.ImagePayload
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// 文档导入与切分。
//
// 支持的输入：
//   - 纯文本类（txt / md / csv / json / 代码文件等）：直接读文本
//   - PDF：Android 没有内置的文本层提取 API（PdfRenderer 只能渲染位图），
//     PDF 要走"渲染成图 → 交给视觉模型识别"这条路。
//     因此需要外部注入 visionExtractor；未注入时 PDF 导入会明确失败而不是静默跳过。

data class IngestResult(val docId: Long, val nChunks: Int)

class DocumentIngestor(
    private val context: Context,
    private val db: KnowledgeDb,
    private val embedder: Embedder,
) {
    // 由上层注入：把一页图片转成文本。通常是调用多模态模型做 OCR/描述。
    var visionExtractor: (suspend (ImagePayload) -> String)? = null

    // PDF 渲染分辨率。再高对 OCR 收益有限，但会显著拖慢推理。
    var pdfRenderWidth: Int = 1024

    suspend fun ingestUri(
        uri: Uri,
        onProgress: (String) -> Unit = {},
    ): Result<IngestResult> = withContext(Dispatchers.IO) {
        runCatching {
            val mime = context.contentResolver.getType(uri) ?: "text/plain"
            val name = queryDisplayName(uri) ?: uri.lastPathSegment ?: "未命名"

            when {
                mime == "application/pdf" || name.endsWith(".pdf", true) ->
                    ingestPdf(uri, name, onProgress)
                mime.startsWith("image/") ->
                    ingestImage(uri, name, onProgress)
                else ->
                    ingestText(name, uri.toString(), readText(uri), onProgress).getOrThrow()
            }
        }
    }

    suspend fun ingestText(
        title: String,
        source: String,
        text: String,
        onProgress: (String) -> Unit = {},
    ): Result<IngestResult> = withContext(Dispatchers.IO) {
        runCatching {
            val chunks = Chunker.split(text)
            if (chunks.isEmpty()) error("文档内容为空")

            val docId = db.insertDocument(title, source, "text/plain")
            var done = 0
            // 分批向量化，避免一次提交过多文本导致 embedding 上下文溢出
            for (batch in chunks.chunked(BATCH)) {
                val vecs = if (embedder.isReady) {
                    runCatching { embedder.embed(batch) }.getOrNull()
                } else null

                batch.forEachIndexed { i, c ->
                    db.insertChunk(docId, done + i, c, vecs?.getOrNull(i))
                }
                done += batch.size
                onProgress("已索引 $done/${chunks.size} 段")
            }
            db.setChunkCount(docId, done)
            db.invalidateVectorCache()
            IngestResult(docId, done)
        }
    }

    private suspend fun ingestPdf(
        uri: Uri,
        name: String,
        onProgress: (String) -> Unit,
    ): IngestResult {
        val extract = visionExtractor
            ?: error("PDF 需要多模态模型做识别，请先加载支持视觉的模型")

        val pfd = context.contentResolver.openFileDescriptor(uri, "r")
            ?: error("无法打开文件")
        val renderer = PdfRenderer(pfd)

        val docId = db.insertDocument(name, uri.toString(), "application/pdf")
        var done = 0

        renderer.use { r ->
            for (i in 0 until r.pageCount) {
                val page = r.openPage(i)
                val scale = pdfRenderWidth.toFloat() / page.width
                val w = pdfRenderWidth
                val h = (page.height * scale).toInt().coerceAtLeast(1)

                val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
                page.render(bmp, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                page.close()
                onProgress("识别第 ${i + 1}/${r.pageCount} 页")

                val payload = bitmapToPayload(bmp)
                bmp.recycle()

                val text = extract(payload).trim()
                if (text.isNotBlank()) {
                    // 一页可能较长，继续切分
                    val chunks = Chunker.split(text)
                    for (batch in chunks.chunked(BATCH)) {
                        val vecs = if (embedder.isReady) {
                            runCatching { embedder.embed(batch) }.getOrNull()
                        } else null
                        batch.forEachIndexed { k, c ->
                            db.insertChunk(docId, done + k, c, vecs?.getOrNull(k))
                        }
                        done += batch.size
                    }
                }
            }
        }

        pfd.close()
        db.setChunkCount(docId, done)
        db.invalidateVectorCache()
        return IngestResult(docId, done)
    }

    private suspend fun ingestImage(
        uri: Uri,
        name: String,
        onProgress: (String) -> Unit,
    ): IngestResult {
        val extract = visionExtractor
            ?: error("图片需要多模态模型做识别，请先加载支持视觉的模型")

        val bmp = android.provider.MediaStore.Images.Media.getBitmap(
            context.contentResolver, uri
        )
        onProgress("识别图片中…")

        val text = extract(bitmapToPayload(bmp))
        bmp.recycle()

        return ingestText(name, uri.toString(), text, onProgress).getOrThrow()
    }

    private fun readText(uri: Uri): String =
        context.contentResolver.openInputStream(uri)?.bufferedReader()?.use { it.readText() }
            ?: error("无法读取文件")

    private fun queryDisplayName(uri: Uri): String? {
        context.contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
            if (idx >= 0 && c.moveToFirst()) return c.getString(idx)
        }
        return null
    }

    companion object {
        private const val BATCH = 16

        fun bitmapToPayload(bmp: Bitmap): ImagePayload {
            val w = bmp.width
            val h = bmp.height
            val pixels = IntArray(w * h)
            bmp.getPixels(pixels, 0, w, 0, 0, w, h)

            val rgb = ByteArray(w * h * 3)
            var p = 0
            for (argb in pixels) {
                rgb[p++] = ((argb shr 16) and 0xFF).toByte()
                rgb[p++] = ((argb shr 8) and 0xFF).toByte()
                rgb[p++] = (argb and 0xFF).toByte()
            }
            return ImagePayload(w, h, rgb)
        }
    }
}

// 文本切分。
//
// 策略：优先按段落边界切，段落过长再退化为按字符切，段间保留重叠。
// 重叠很关键 —— 硬切断会把一句话的关键主语和谓语拆到两个片段里，
// 两边单独看都不完整，检索命中率会明显下降。
object Chunker {
    private const val MAX_CHARS = 700
    private const val OVERLAP = 80

    fun split(text: String, maxChars: Int = MAX_CHARS, overlap: Int = OVERLAP): List<String> {
        val clean = text.replace("\r\n", "\n").trim()
        if (clean.isBlank()) return emptyList()

        val paragraphs = clean.split(Regex("\n{2,}"))
            .map { it.trim() }
            .filter { it.isNotEmpty() }

        val out = mutableListOf<String>()
        val buf = StringBuilder()

        fun flush() {
            if (buf.isNotBlank()) out += buf.toString().trim()
            buf.clear()
        }

        for (para in paragraphs) {
            if (para.length <= maxChars) {
                if (buf.length + para.length + 2 > maxChars) flush()
                if (buf.isNotEmpty()) buf.append('\n')
                buf.append(para)
            } else {
                flush()
                // 长段落按字符滑窗切分
                var start = 0
                while (start < para.length) {
                    val end = (start + maxChars).coerceAtMost(para.length)
                    out += para.substring(start, end).trim()
                    if (end >= para.length) break
                    start = (end - overlap).coerceAtLeast(start + 1)
                }
            }
        }
        flush()

        return out.filter { it.length >= 20 }   // 丢弃过碎的片段
            .ifEmpty { listOf(clean.take(maxChars)) }
    }
}
