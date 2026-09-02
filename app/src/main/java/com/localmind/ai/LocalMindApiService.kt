package com.localmind.ai

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.RemoteException
import androidx.core.app.NotificationCompat
import com.localmind.ai.engine.AppServices
import com.localmind.ai.engine.GenStats
import com.localmind.ai.engine.ImagePayload
import com.localmind.ai.engine.NativeEngine
import com.localmind.ai.engine.SamplerConfig
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.json.JSONArray
import org.json.JSONObject

// 对外 API 服务（AIDL）。
//
// 以「前台服务」形式运行：模型推理是长耗时的前台任务，必须用前台服务保活，
// 否则 Android 会在应用退到后台后回收进程，外部 App 的调用就会失败。
//
// 模型是共享的：本 App 的 UI 与外部 App 调用的是同一份 loaded 模型，
// 通过 ModelRuntime 的 Mutex 串行化，不会互相踩踏。
class LocalMindApiService : Service() {

    private lateinit var svc: AppServices
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private val binder = object : ILocalMindApi.Stub() {

        override fun isReady(): Boolean = svc.runtime.isLoaded

        override fun modelId(): String = svc.currentModelId

        override fun generate(prompt: String?, paramsJson: String?): String {
            if (prompt.isNullOrEmpty()) return ""
            val sp = parseSampler(paramsJson)
            val sb = StringBuilder()
            val stats = runBlocking {
                svc.runtime.generate(prompt, null, sp) { sb.append(it); true }
            }
            return if (stats.error != null) "ERROR: ${stats.error}" else sb.toString()
        }

        override fun generateStream(
            prompt: String?,
            paramsJson: String?,
            cb: ILocalMindStreamCallback?,
        ) {
            if (prompt.isNullOrEmpty() || cb == null) return
            val sp = parseSampler(paramsJson)
            scope.launch {
                svc.runtime.generate(prompt, null, sp) { piece ->
                    runCatching { cb.onToken(piece) }.isSuccess
                }.also { stats ->
                    runCatching { cb.onDone(statsToJson(stats)) }
                }
            }
        }

        override fun cancel() {
            svc.runtime.cancel()
        }

        override fun applyTemplate(messagesJson: String?): String {
            if (messagesJson.isNullOrEmpty()) return ""
            val msgs = parseMessages(messagesJson) ?: return ""
            return svc.runtime.renderTemplate(msgs) ?: ""
        }

        override fun embed(text: String?): FloatArray {
            if (text.isNullOrEmpty() || !svc.embedder.isReady) return FloatArray(0)
            return svc.embedder.embed(listOf(text))?.firstOrNull() ?: FloatArray(0)
        }

        override fun ragSearch(query: String?, topK: Int): String {
            if (query.isNullOrEmpty()) return "[]"
            val hits = runBlocking { svc.retriever.retrieve(query, topK.coerceAtLeast(1)) }
            val arr = JSONArray()
            hits.forEach { (chunk) ->
                arr.put(JSONObject().apply {
                    put("title", chunk.docTitle)
                    put("content", chunk.content)
                })
            }
            return arr.toString()
        }

        override fun runtimeInfo(): String =
            NativeEngine.probe().let {
                JSONObject().apply {
                    put("socModel", it.socModel)
                    put("gpuName", it.gpuName)
                    put("nCores", it.nCores)
                    put("totalRamMb", it.totalRamMb)
                    put("openclAvailable", it.openclAvailable)
                }.toString()
            }

        override fun httpEndpoint(): String = svc.httpEndpoint()
    }

    override fun onCreate() {
        super.onCreate()
        svc = (application as LocalMindApplication).services
        ensureChannel()
        startForeground(NOTIF_ID, buildNotification("LocalMind 本地 AI 已就绪"))
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onDestroy() {
        super.onDestroy()
        scope.cancel()
    }

    // -----------------------------------------------------------------
    private fun parseSampler(json: String?): SamplerConfig {
        if (json.isNullOrEmpty()) return SamplerConfig()
        return runCatching {
            val j = JSONObject(json)
            SamplerConfig(
                temp = j.optDouble("temp", 0.7).toFloat(),
                topP = j.optDouble("topP", 0.9).toFloat(),
                topK = j.optDouble("topK", 40.0).toFloat(),
                minP = j.optDouble("minP", 0.05).toFloat(),
                repeatPenalty = j.optDouble("repeatPenalty", 1.1).toFloat(),
                repeatLastN = j.optInt("repeatLastN", 64),
                nPredict = j.optInt("nPredict", -1),
                seed = j.optInt("seed", 0),
            )
        }.getOrDefault(SamplerConfig())
    }

    private fun parseMessages(json: String): List<Pair<String, String>>? = runCatching {
        val arr = JSONArray(json)
        val out = mutableListOf<Pair<String, String>>()
        for (i in 0 until arr.length()) {
            val m = arr.optJSONObject(i) ?: continue
            out += (m.optString("role", "user") to m.optString("content", ""))
        }
        out
    }.getOrNull()

    private fun statsToJson(s: GenStats): String = JSONObject().apply {
        put("stopReason", s.stopReason)
        put("nGenTokens", s.nGenTokens)
        put("genTps", s.genTps)
        put("nDrafted", s.nDrafted)
        put("nAccepted", s.nAccepted)
        put("acceptRate", s.acceptRate)
    }.toString()

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val mgr = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            if (mgr.getNotificationChannel(CHANNEL_ID) == null) {
                val ch = NotificationChannel(
                    CHANNEL_ID, "LocalMind 本地 AI",
                    NotificationManager.IMPORTANCE_LOW,
                )
                ch.setShowBadge(false)
                mgr.createNotificationChannel(ch)
            }
        }
    }

    private fun buildNotification(text: String): Notification {
        val intent = packageManager.getLaunchIntentForPackage(packageName)
        val pi = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("LocalMind 本地 AI")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pi)
            .setOngoing(true)
            .build()
    }

    companion object {
        private const val CHANNEL_ID = "localmind_api"
        private const val NOTIF_ID = 1001
    }
}
