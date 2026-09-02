package com.localmind.ai.api

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.net.InetAddress
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.nio.charset.StandardCharsets

// 极简 HTTP/1.1 服务器。
//
// 这里刻意不引入 Ktor / NanoHTTPD 之类的库：
// 本应用只需要暴露几个固定路由，引入一个 Web 框架会带来几百 KB 体积
// 和一堆传递依赖，收益不成比例。手写约 200 行即可覆盖需求。
//
// 只实现必要的能力：
//   - 单连接单请求（响应后关闭，不做 keep-alive 复用）
//   - 支持 Content-Length 请求体
//   - 支持 chunked / SSE 流式响应
//   - 默认只监听 127.0.0.1，避免把本地模型暴露到局域网

data class HttpRequest(
    val method: String,
    val path: String,
    val query: Map<String, String>,
    val headers: Map<String, String>,
    val body: ByteArray,
) {
    fun bodyText(): String = body.toString(StandardCharsets.UTF_8)
    fun header(name: String): String? =
        headers.entries.firstOrNull { it.key.equals(name, ignoreCase = true) }?.value
}

interface HttpHandler {
    // 返回 true 表示已完整处理并写出响应；false 表示路由未命中（返回 404）
    suspend fun handle(req: HttpRequest, out: ResponseWriter): Boolean
}

interface ResponseWriter {
    fun status(code: Int, reason: String)
    fun header(name: String, value: String)
    fun write(bytes: ByteArray)
    fun writeText(text: String) = write(text.toByteArray(StandardCharsets.UTF_8))
    fun flush()
    fun endHeaders()
}

class HttpServer(
    private val port: Int,
    private val handler: HttpHandler,
    private val bindHost: String = "127.0.0.1",
    private val apiKey: String = "",   // 非空时要求 Authorization: Bearer <key>
) {
    private var serverSocket: ServerSocket? = null
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    val isRunning: Boolean get() = serverSocket?.isClosed == false

    @Volatile var lastError: String? = null
        private set

    fun start(): Boolean {
        if (isRunning) return true
        return try {
            val addr = InetAddress.getByName(bindHost)
            serverSocket = ServerSocket(port, 64, addr).also {
                it.reuseAddress = true
            }
            scope.launch { acceptLoop() }
            true
        } catch (e: Exception) {
            lastError = e.message
            false
        }
    }

    fun stop() {
        runCatching { serverSocket?.close() }
        serverSocket = null
    }

    // 从字节流逐字节读一行（兼容 \n / \r\n），不破坏后续二进制的 body 字节。
    private fun readLine(input: InputStream): String? {
        val sb = StringBuilder()
        while (true) {
            val b = input.read()
            if (b == -1) return if (sb.isEmpty()) null else sb.toString()
            if (b == '\n'.code) return sb.toString()
            if (b != '\r'.code) sb.append(b.toChar())
        }
    }

    private suspend fun acceptLoop() {
        val ss = serverSocket ?: return
        while (scope.isActive && !ss.isClosed) {
            val client = try {
                withContext(Dispatchers.IO) { ss.accept() }
            } catch (e: SocketException) {
                break
            } catch (e: Exception) {
                if (ss.isClosed) break else continue
            }
            scope.launch { handleClient(client) }
        }
    }

    private suspend fun handleClient(client: Socket) {
        try {
            withContext(Dispatchers.IO) { client.soTimeout = 0 }
            val input = BufferedInputStream(client.getInputStream(), 32 * 1024)
            val rawOut = BufferedOutputStream(client.getOutputStream(), 32 * 1024)

            val requestLine = readLine(input) ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return

            val method = parts[0]
            val target = parts[1]

            // 解析 header
            val headers = HashMap<String, String>()
            var line: String?
            while (readLine(input).also { line = it } != null) {
                if (line.isNullOrEmpty()) break
                val idx = line!!.indexOf(':')
                if (idx > 0) {
                    headers[line!!.substring(0, idx).trim()] = line!!.substring(idx + 1).trim()
                }
            }

            // 解析 body
            val body = if (method == "POST" || method == "PUT" || method == "PATCH") {
                val len = headers["Content-Length"]?.toIntOrNull() ?: 0
                if (len > 0) {
                    val buf = ByteArray(len)
                    var read = 0
                    while (read < len) {
                        val n = input.read(buf, read, len - read)
                        if (n < 0) break
                        read += n
                    }
                    buf.copyOf(read)
                } else ByteArray(0)
            } else ByteArray(0)

            val (path, query) = parseTarget(target)
            val req = HttpRequest(method, path, query, headers, body)

            val writer = SocketResponseWriter(rawOut)

            // API Key 校验
            if (apiKey.isNotEmpty()) {
                val auth = req.header("Authorization")
                val ok = auth?.removePrefix("Bearer ")?.trim() == apiKey
                if (!ok) {
                    writer.status(401, "Unauthorized")
                    writer.header("Content-Type", "application/json")
                    writer.endHeaders()
                    writer.writeText("""{"error":{"message":"invalid api key"}}""")
                    writer.flush()
                    return
                }
            }

            val handled = runCatching { handler.handle(req, writer) }.getOrDefault(false)
            if (!handled) {
                writer.status(404, "Not Found")
                writer.header("Content-Type", "application/json")
                writer.endHeaders()
                writer.writeText("""{"error":{"message":"not found: $path"}}""")
                writer.flush()
            }
            writer.flush()
        } catch (e: Exception) {
            // 客户端断开连接是常态，不需要记录
        } finally {
            runCatching { client.close() }
        }
    }

    private fun parseTarget(target: String): Pair<String, Map<String, String>> {
        val idx = target.indexOf('?')
        if (idx < 0) return target to emptyMap()
        val path = target.substring(0, idx)
        val query = target.substring(idx + 1).split("&")
            .mapNotNull {
                val kv = it.split("=", limit = 2)
                if (kv.size == 2) kv[0] to kv[1] else null
            }.toMap()
        return path to query
    }

    private class SocketResponseWriter(
        private val out: OutputStream,
    ) : ResponseWriter {
        private var headersSent = false
        private var statusCode = 200
        private var reason = "OK"
        private val headers = LinkedHashMap<String, String>()
        private val lock = Any()

        override fun status(code: Int, reason: String) {
            statusCode = code; this.reason = reason
        }

        override fun header(name: String, value: String) { headers[name] = value }

        override fun endHeaders() {
            synchronized(lock) {
                if (headersSent) return
                headersSent = true
                val sb = StringBuilder()
                sb.append("HTTP/1.1 ").append(statusCode).append(' ').append(reason).append("\r\n")
                if (!headers.containsKey("Content-Type"))
                    sb.append("Content-Type: application/json\r\n")
                headers.forEach { (k, v) -> sb.append(k).append(": ").append(v).append("\r\n") }
                sb.append("Connection: close\r\n\r\n")
                out.write(sb.toString().toByteArray(StandardCharsets.ISO_8859_1))
            }
        }

        // 流式场景下 native 线程会直接调用这里，必须加锁
        override fun write(bytes: ByteArray) {
            synchronized(lock) {
                endHeaders()
                out.write(bytes)
            }
        }

        override fun flush() {
            synchronized(lock) { out.flush() }
        }
    }
}
