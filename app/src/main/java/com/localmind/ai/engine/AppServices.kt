package com.localmind.ai.engine

import android.content.Context
import com.localmind.ai.api.HttpServer
import com.localmind.ai.api.InferenceBackend
import com.localmind.ai.api.OpenAiRoutes
import com.localmind.ai.rag.Embedder
import com.localmind.ai.rag.HybridRetriever
import com.localmind.ai.rag.KnowledgeDb
import kotlinx.coroutines.runBlocking

// 应用级服务容器。
//
// 持有全部重量级单例，并负责把 HTTP 服务、AIDL 服务、UI 接到同一个后端上。
// 在 LocalMindApplication.onCreate 里创建并 init()。
class AppServices(private val ctx: Context) {

    val runtime = ModelRuntime()
    val embedder = Embedder()
    val db = KnowledgeDb.get(ctx)
    val retriever = HybridRetriever(db, embedder)

    var currentModelId = ""
        private set

    // 设备画像。用 lazy 延迟到 so 加载完成后再探测。
    val profile: DeviceProfile by lazy {
        NativeEngine.ensureLoaded()
        NativeEngine.probe().let {
            DeviceProfile(it.socModel, it.gpuName, it.nCores, it.totalRamMb, it.openclAvailable)
        }
    }

    val backend: InferenceBackend =
        LocalMindBackend(runtime, embedder, retriever) { currentModelId }

    // HTTP 服务实例（未启动时为 null）
    var http: HttpServer? = null
        private set
    private var httpHost = "127.0.0.1"
    private var httpPort = 0

    fun init() {
        runtime.init()
    }

    // ---------------------------------------------------------------------
    // 模型加载
    // ---------------------------------------------------------------------

    // 给定模型路径与性能档位，用骁龙调优器生成参数并加载。
    // modelBytes 用于内存预算估算；传 0 表示未知（按路径无法获知时）。
    suspend fun loadModel(
        path: String,
        mmproj: String = "",
        preset: PerfPreset = PerfPreset.BALANCED,
        modelBytes: Long = 0L,
        draftPath: String = "",
    ): Result<ModelInfo> {
        val cfg = SnapdragonTuner.buildConfig(profile, path, mmproj, modelBytes, preset).let {
            if (draftPath.isNotEmpty())
                it.copy(draftPath = draftPath, specMode = SpecMode.DRAFT_MODEL)
            else it
        }
        return runtime.load(cfg).onSuccess { currentModelId = fileName(path) }
    }

    fun unloadModel() {
        runBlocking { runtime.unload() }
        currentModelId = ""
    }

    // ---------------------------------------------------------------------
    // 本地 API（HTTP）
    // ---------------------------------------------------------------------

    // 默认只监听 127.0.0.1：本地模型不该默认暴露到局域网。
    // 确需局域网访问时，把 host 改成 "0.0.0.0" 并加上 apiKey。
    fun startHttp(port: Int = 8080, host: String = "127.0.0.1", apiKey: String = ""): Boolean {
        if (http?.isRunning == true) return true
        val s = HttpServer(port, OpenAiRoutes(backend), host, apiKey)
        if (s.start()) {
            http = s
            httpHost = host
            httpPort = port
            LogFile.i("AppServices", "HttpServer started: http://$host:$port")
            return true
        }
        LogFile.e("AppServices", "HttpServer start FAILED: ${s.lastError} (port=$port host=$host)")
        return false
    }

    fun stopHttp() {
        http?.stop()
        http = null
        httpPort = 0
    }

    fun isHttpRunning(): Boolean = http?.isRunning == true

    fun httpEndpoint(): String =
        if (isHttpRunning()) "http://$httpHost:$httpPort" else ""

    // ---------------------------------------------------------------------
    private fun fileName(p: String): String =
        p.substringAfterLast('/').substringAfterLast('\\')
}
