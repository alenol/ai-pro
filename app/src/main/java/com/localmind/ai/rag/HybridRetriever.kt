package com.localmind.ai.rag

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

// 混合检索：BM25 关键词 + 向量语义，用 RRF 融合。
//
// 为什么两套都要：
//   - BM25 擅长精确匹配：型号、函数名、专有名词、代码符号。
//     这些词在语义空间里往往彼此很近，向量检索容易混淆。
//   - 向量擅长同义泛化：用户问"怎么省电"，文档写"降低功耗"，关键词完全不重合。
//
// 两者单独用都会在另一类查询上翻车，融合是最省事也最稳的做法。

data class RetrievedChunk(
    val chunk: ChunkRow,
    val score: Float,
    val bm25Rank: Int?,   // null 表示该路没召回
    val vecRank: Int?,
) {
    val content: String get() = chunk.content
    val docTitle: String get() = chunk.docTitle
}

class HybridRetriever(
    private val db: KnowledgeDb,
    private val embedder: Embedder,
) {
    // RRF 常数。60 是原始论文的经验值，调大倾向于平等对待两路结果。
    private val rrfK = 60f

    // 权重：关键词与语义的相对重要性。
    // 中文场景下 BM25 按字切分噪音偏多，默认给语义稍高权重。
    var bm25Weight: Float = 0.8f
    var vectorWeight: Float = 1.0f

    suspend fun retrieve(query: String, topK: Int = 5): List<RetrievedChunk> =
        withContext(Dispatchers.IO) {
            val poolSize = (topK * 6).coerceAtLeast(20)

            val bm25 = runCatching { db.searchFts(query, poolSize) }.getOrDefault(emptyList())

            val vec: List<ScoredChunk> = if (embedder.isReady) {
                val q = runCatching { embedder.embedOne(query) }.getOrNull()
                if (q != null) {
                    runCatching { db.searchVector(q, poolSize) }.getOrDefault(emptyList())
                } else emptyList()
            } else emptyList()

            fuse(bm25, vec).take(topK)
        }

    private fun fuse(
        bm25: List<ScoredChunk>,
        vec: List<ScoredChunk>,
    ): List<RetrievedChunk> {
        val bm25Rank = HashMap<Long, Int>()
        bm25.forEachIndexed { i, s -> bm25Rank[s.chunk.id] = i + 1 }

        val vecRank = HashMap<Long, Int>()
        vec.forEachIndexed { i, s -> vecRank[s.chunk.id] = i + 1 }

        val byId = HashMap<Long, ChunkRow>()
        (bm25 + vec).forEach { byId[it.chunk.id] = it.chunk }

        val scored = byId.map { (id, chunk) ->
            var score = 0f
            bm25Rank[id]?.let { score += bm25Weight * (1f / (rrfK + it)) }
            vecRank[id]?.let  { score += vectorWeight * (1f / (rrfK + it)) }
            RetrievedChunk(chunk, score, bm25Rank[id], vecRank[id])
        }
        return scored.sortedByDescending { it.score }
    }

    // 组装进 prompt 的上下文。长度做上限截断，避免吃掉生成预算。
    fun buildContext(hits: List<RetrievedChunk>, maxChars: Int = 6000): String {
        if (hits.isEmpty()) return ""
        val sb = StringBuilder()
        var used = 0
        hits.forEachIndexed { i, hit ->
            val block = "[$i] 来源：${hit.docTitle}\n${hit.content}\n"
            if (used + block.length > maxChars) return@forEachIndexed
            sb.append(block).append('\n')
            used += block.length
        }
        return sb.toString()
    }
}
