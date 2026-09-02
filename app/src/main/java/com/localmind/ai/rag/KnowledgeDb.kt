package com.localmind.ai.rag

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.nio.ByteBuffer
import java.nio.ByteOrder

// 本地知识库存储。
//
// 这里刻意不使用 Room：知识库要用到 FTS5 虚拟表和 BLOB 向量，
// 直接写 SQL 比套一层 ORM 更直接，也少一个构建期代码生成环节。
//
// 表结构：
//   document  — 一个导入的文档
//   chunk     — 切分后的片段，含向量
//   chunk_fts — FTS5 虚拟表，负责 BM25 关键词检索

data class DocRow(
    val id: Long,
    val title: String,
    val source: String,
    val mime: String,
    val createdAt: Long,
    val nChunks: Int,
)

data class ChunkRow(
    val id: Long,
    val docId: Long,
    val idx: Int,
    val content: String,
    val docTitle: String,
)

data class ScoredChunk(
    val chunk: ChunkRow,
    val score: Float,
)

class KnowledgeDb private constructor(context: Context) :
    SQLiteOpenHelper(context, DB_NAME, null, DB_VERSION) {

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """
            CREATE TABLE document (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                title      TEXT NOT NULL,
                source     TEXT NOT NULL,
                mime       TEXT NOT NULL DEFAULT 'text/plain',
                created_at INTEGER NOT NULL,
                n_chunks   INTEGER NOT NULL DEFAULT 0
            )
            """.trimIndent()
        )

        db.execSQL(
            """
            CREATE TABLE chunk (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id    INTEGER NOT NULL REFERENCES document(id) ON DELETE CASCADE,
                idx       INTEGER NOT NULL,
                content   TEXT NOT NULL,
                embedding BLOB
            )
            """.trimIndent()
        )
        db.execSQL("CREATE INDEX idx_chunk_doc ON chunk(doc_id)")

        // 索引用的内容单独存一份预处理后的 token 串（见 toFtsTokens 的说明）
        db.execSQL(
            """
            CREATE VIRTUAL TABLE chunk_fts USING fts5(
                content,
                chunk_id UNINDEXED,
                doc_id   UNINDEXED,
                tokenize = 'unicode61 remove_diacritics 2'
            )
            """.trimIndent()
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldV: Int, newV: Int) {
        db.execSQL("DROP TABLE IF EXISTS chunk_fts")
        db.execSQL("DROP TABLE IF EXISTS chunk")
        db.execSQL("DROP TABLE IF EXISTS document")
        onCreate(db)
    }

    override fun onConfigure(db: SQLiteDatabase) {
        db.setForeignKeyConstraintsEnabled(true)
    }

    // ---------------------------------------------------------------------
    // 写入
    // ---------------------------------------------------------------------

    fun insertDocument(title: String, source: String, mime: String): Long {
        val cv = android.content.ContentValues().apply {
            put("title", title)
            put("source", source)
            put("mime", mime)
            put("created_at", System.currentTimeMillis())
            put("n_chunks", 0)
        }
        return writableDatabase.insert("document", null, cv)
    }

    fun insertChunk(docId: Long, idx: Int, content: String, embedding: FloatArray?) {
        val db = writableDatabase
        val cv = android.content.ContentValues().apply {
            put("doc_id", docId)
            put("idx", idx)
            put("content", content)
            if (embedding != null) put("embedding", floatsToBlob(embedding))
        }
        val cid = db.insert("chunk", null, cv)

        val fts = android.content.ContentValues().apply {
            put("content", toFtsTokens(content))
            put("chunk_id", cid)
            put("doc_id", docId)
        }
        db.insert("chunk_fts", null, fts)
    }

    fun setChunkCount(docId: Long, n: Int) {
        val cv = android.content.ContentValues().apply { put("n_chunks", n) }
        writableDatabase.update("document", cv, "id = ?", arrayOf(docId.toString()))
    }

    fun deleteDocument(docId: Long) {
        val db = writableDatabase
        db.delete("chunk_fts", "doc_id = ?", arrayOf(docId.toString()))
        db.delete("chunk", "doc_id = ?", arrayOf(docId.toString()))
        db.delete("document", "id = ?", arrayOf(docId.toString()))
    }

    fun clearAll() {
        val db = writableDatabase
        db.delete("chunk_fts", null, null)
        db.delete("chunk", null, null)
        db.delete("document", null, null)
    }

    fun listDocuments(): List<DocRow> {
        val out = mutableListOf<DocRow>()
        readableDatabase.query("document", null, null, null, null, null, "created_at DESC")
            .use { c ->
                while (c.moveToNext()) {
                    out += DocRow(
                        id = c.getLong(c.getColumnIndexOrThrow("id")),
                        title = c.getString(c.getColumnIndexOrThrow("title")),
                        source = c.getString(c.getColumnIndexOrThrow("source")),
                        mime = c.getString(c.getColumnIndexOrThrow("mime")),
                        createdAt = c.getLong(c.getColumnIndexOrThrow("created_at")),
                        nChunks = c.getInt(c.getColumnIndexOrThrow("n_chunks")),
                    )
                }
            }
        return out
    }

    // ---------------------------------------------------------------------
    // 关键词检索（BM25）
    // ---------------------------------------------------------------------

    fun searchFts(query: String, limit: Int): List<ScoredChunk> {
        val q = toFtsTokens(query).trim()
        if (q.isEmpty()) return emptyList()

        // 把用户输入的 token 串转成 FTS5 的 OR 查询，BM25 会自动排序
        val terms = q.split(Regex("\\s+")).filter { it.length >= 1 }.take(24)
        if (terms.isEmpty()) return emptyList()
        val match = terms.joinToString(" OR ") { "\"$it\"" }

        val sql = """
            SELECT c.id AS cid, c.doc_id, c.idx, c.content, d.title,
                   bm25(chunk_fts) AS score
            FROM chunk_fts f
            JOIN chunk c ON c.id = f.chunk_id
            JOIN document d ON d.id = c.doc_id
            WHERE chunk_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """.trimIndent()

        val out = mutableListOf<ScoredChunk>()
        readableDatabase.rawQuery(sql, arrayOf(match, limit.toString())).use { c ->
            while (c.moveToNext()) {
                out += ScoredChunk(
                    chunk = ChunkRow(
                        id = c.getLong(c.getColumnIndexOrThrow("cid")),
                        docId = c.getLong(c.getColumnIndexOrThrow("doc_id")),
                        idx = c.getInt(c.getColumnIndexOrThrow("idx")),
                        content = c.getString(c.getColumnIndexOrThrow("content")),
                        docTitle = c.getString(c.getColumnIndexOrThrow("title")),
                    ),
                    // bm25() 返回负值，越小越相关，这里取反转成"越大越好"
                    score = -c.getFloat(c.getColumnIndexOrThrow("score")),
                )
            }
        }
        return out
    }

    // ---------------------------------------------------------------------
    // 向量检索
    // ---------------------------------------------------------------------

    // 向量在内存中做暴力点积。
    // 手机上的知识库规模通常在万级片段以内，1024 维下单次全量扫描
    // 约几千万次浮点运算，耗时在几十毫秒量级，足够快。
    private var vectorCache: List<Pair<Long, FloatArray>>? = null

    fun invalidateVectorCache() { vectorCache = null }

    private fun loadVectors(): List<Pair<Long, FloatArray>> {
        vectorCache?.let { return it }
        val out = mutableListOf<Pair<Long, FloatArray>>()
        readableDatabase.query(
            "chunk", arrayOf("id", "embedding"),
            "embedding IS NOT NULL", null, null, null, null
        ).use { c ->
            while (c.moveToNext()) {
                val blob = c.getBlob(c.getColumnIndexOrThrow("embedding")) ?: continue
                out += c.getLong(c.getColumnIndexOrThrow("id")) to blobToFloats(blob)
            }
        }
        vectorCache = out
        return out
    }

    // 余弦相似度。向量在写入前已做 L2 归一化，所以点积即余弦。
    fun searchVector(queryVec: FloatArray, limit: Int): List<ScoredChunk> {
        val vecs = loadVectors()
        if (vecs.isEmpty()) return emptyList()

        val scored = ArrayList<Pair<Long, Float>>(vecs.size)
        for ((id, v) in vecs) {
            if (v.size != queryVec.size) continue
            var dot = 0f
            var i = 0
            while (i < v.size) { dot += v[i] * queryVec[i]; i++ }
            scored += id to dot
        }
        scored.sortByDescending { it.second }

        val top = scored.take(limit)
        val rows = fetchChunksByIds(top.map { it.first })
        val scoreById = top.toMap()

        return rows.mapNotNull { row ->
            val s = scoreById[row.id] ?: return@mapNotNull null
            ScoredChunk(row, s)
        }
    }

    private fun fetchChunksByIds(ids: List<Long>): List<ChunkRow> {
        if (ids.isEmpty()) return emptyList()
        val placeholders = ids.joinToString(",") { "?" }
        val sql = """
            SELECT c.id, c.doc_id, c.idx, c.content, d.title
            FROM chunk c JOIN document d ON d.id = c.doc_id
            WHERE c.id IN ($placeholders)
        """.trimIndent()
        val args = ids.map { it.toString() }.toTypedArray()

        val out = mutableListOf<ChunkRow>()
        readableDatabase.rawQuery(sql, args).use { c ->
            while (c.moveToNext()) {
                out += ChunkRow(
                    id = c.getLong(c.getColumnIndexOrThrow("id")),
                    docId = c.getLong(c.getColumnIndexOrThrow("doc_id")),
                    idx = c.getInt(c.getColumnIndexOrThrow("idx")),
                    content = c.getString(c.getColumnIndexOrThrow("content")),
                    docTitle = c.getString(c.getColumnIndexOrThrow("title")),
                )
            }
        }
        return out
    }

    fun chunkCount(): Int {
        readableDatabase.rawQuery("SELECT COUNT(*) FROM chunk", null).use { c ->
            return if (c.moveToFirst()) c.getInt(0) else 0
        }
    }

    companion object {
        const val DB_NAME = "localmind_knowledge.db"
        const val DB_VERSION = 1

        @Volatile private var INSTANCE: KnowledgeDb? = null

        fun get(context: Context): KnowledgeDb =
            INSTANCE ?: synchronized(this) {
                INSTANCE ?: KnowledgeDb(context.applicationContext).also { INSTANCE = it }
            }

        // FTS5 的 unicode61 分词器按空格与标点切词，对中文会整段当成一个 token，
        // 导致检索完全失效。这里做一次预处理：
        //   - CJK 字符逐字拆开（用空格分隔）
        //   - 拉丁字母与数字保持原词
        // 这样 BM25 对中文查询依然可用：高频虚词会被 BM25 自然降权。
        //
        // 如果需要更好的中文召回，可以改成 bigram（相邻两字一组），
        // 代价是索引体积翻倍。
        fun toFtsTokens(text: String): String {
            val sb = StringBuilder(text.length * 2)
            var i = 0
            while (i < text.length) {
                val cp = text.codePointAt(i)
                val charCount = Character.charCount(cp)
                if (isCjk(cp)) {
                    if (sb.isNotEmpty() && sb.last() != ' ') sb.append(' ')
                    sb.appendCodePoint(cp)
                    sb.append(' ')
                } else {
                    sb.appendCodePoint(cp)
                }
                i += charCount
            }
            return sb.toString()
        }

        private fun isCjk(cp: Int): Boolean =
            (cp in 0x4E00..0x9FFF) ||     // CJK 统一表意文字
            (cp in 0x3400..0x4DBF) ||     // 扩展 A
            (cp in 0x3040..0x30FF) ||     // 日文假名
            (cp in 0xAC00..0xD7AF)        // 韩文音节

        fun floatsToBlob(v: FloatArray): ByteArray {
            val buf = ByteBuffer.allocate(v.size * 4).order(ByteOrder.nativeOrder())
            buf.asFloatBuffer().put(v)
            return buf.array()
        }

        fun blobToFloats(b: ByteArray): FloatArray {
            val out = FloatArray(b.size / 4)
            ByteBuffer.wrap(b).order(ByteOrder.nativeOrder()).asFloatBuffer().get(out)
            return out
        }
    }
}
