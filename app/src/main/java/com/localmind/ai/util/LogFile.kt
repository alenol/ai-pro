package com.localmind.ai.util

import android.content.ContentValues
import android.content.Context
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.OutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 详细日志输出：把运行日志写到公共下载目录，无需 adb 也能从文件管理器读取。
 *
 * 目录：Download/localmodel/
 *   文件：localmind_yyyyMMdd_HHmmss.log（每次启动一个新文件）
 *   - Android 10+：通过 MediaStore.Downloads 写入，不需要存储权限
 *   - Android 9- ：回退到应用专属外部目录（免权限）
 *
 * 依赖 BuildConfig.DEBUG_LOG（构建时传 -Plocalmind.debugLog=true 开启，见 build.gradle.kts）。
 * 额外安装全局未捕获异常处理器，把崩溃堆栈也写进日志文件。
 */
object LogFile {

    private val enabled = AtomicBoolean(false)

    // 用户全局崩溃处理器（用于链式转发，保证系统崩溃提示仍会出现）
    private val systemHandler: Thread.UncaughtExceptionHandler =
        Thread.getDefaultUncaughtExceptionHandler()
            ?: Thread.UncaughtExceptionHandler { t, e ->
                Log.e("CRASH", "uncaught in ${t.name}", e)
            }

    @Volatile
    private var writer: OutputStream? = null
    @Volatile
    private var fallbackToLogcat = false

    // 目标目录名（跟随用户习惯，放在公共下载目录下）
    private const val DIR = "localmodel"
    private const val TAG = "LogFile"

    fun init(context: Context, debug: Boolean) {
        if (!debug) return
        enabled.set(true)
        try {
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val fileName = "localmind_$stamp.log"
            val header = buildHeader(context)

            if (Build.VERSION.SDK_INT >= 29) {
                val resolver = context.contentResolver
                val values = ContentValues().apply {
                    put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
                    put(MediaStore.MediaColumns.RELATIVE_PATH, "${Environment.DIRECTORY_DOWNLOADS}/$DIR")
                    put(MediaStore.MediaColumns.MIME_TYPE, "text/plain")
                    put(MediaStore.MediaColumns.IS_PENDING, 1)
                }
                val uri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                if (uri != null) {
                    val os = resolver.openOutputStream(uri) ?: throw IOException("openOutputStream -> null")
                    os.write(header.toByteArray())
                    os.flush()
                    // 写完内容再取消 pending，让文件立即可见
                    values.clear()
                    values.put(MediaStore.MediaColumns.IS_PENDING, 0)
                    resolver.update(uri, values, null, null)
                    writer = os
                }
            }

            if (writer == null) {
                // 低版本回退：应用专属外部目录，免权限
                val dir = context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS)
                    ?.let { File(it, DIR) }
                if (dir != null && (dir.exists() || dir.mkdirs())) {
                    val os = FileOutputStream(File(dir, fileName))
                    os.write(header.toByteArray())
                    os.flush()
                    writer = os
                }
            }
        } catch (t: Throwable) {
            // 日志子系统自身出错不能拖垮 App：退化为 logcat
            fallbackToLogcat = true
            Log.w(TAG, "init failed, fallback to logcat", t)
        }

        installCrashHandler()
    }

    private fun buildHeader(context: Context): String {
        val sb = StringBuilder()
        sb.append("=== LocalMind detailed log ===\n")
        sb.append("time: ").append(now()).append('\n')
        sb.append("brand: ").append(Build.BRAND)
            .append("  device: ").append(Build.DEVICE)
            .append("  model: ").append(Build.MODEL).append('\n')
        sb.append("android: ").append(Build.VERSION.RELEASE)
            .append(" (SDK ").append(Build.VERSION.SDK_INT)
            .append(", ").append(Build.VERSION.CODENAME).append(")\n")
        sb.append("fingerprint: ").append(Build.FINGERPRINT).append('\n')
        sb.append("abis: ").append(Build.SUPPORTED_ABIS.joinToString(", ")).append('\n')
        try {
            val pkg = context.packageManager.getPackageInfo(context.packageName, 0)
            sb.append("app: ").append(context.packageName)
                .append(" v").append(pkg.versionName)
                .append(" (").append(pkg.versionCode).append(")\n")
        } catch (_: Throwable) {
        }
        sb.append("detail log: enabled\n")
        sb.append("--------------------------------------------------\n\n")
        return sb.toString()
    }

    private fun installCrashHandler() {
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                e("CRASH", "fatal exception in thread '${thread.name}'", throwable)
                flush()
            } catch (_: Throwable) {
            }
            try {
                systemHandler.uncaughtException(thread, throwable)
            } catch (_: Throwable) {
            }
        }
    }

    // ------------------------------------------------------------------
    // 对外日志 API
    // ------------------------------------------------------------------
    fun i(tag: String, msg: String) = write('I', tag, msg)

    fun w(tag: String, msg: String) = write('W', tag, msg)

    fun e(tag: String, msg: String, t: Throwable? = null) {
        write('E', tag, msg)
        t?.let { write('E', tag, Log.getStackTraceString(it)) }
    }

    private fun write(level: Char, tag: String, msg: String) {
        if (!enabled.get()) return
        val line = "${now()} $level/$tag: $msg\n"
        val out = writer
        if (out != null) {
            try {
                out.write(line.toByteArray())
                out.flush()
            } catch (_: Throwable) {
            }
        }
        if (fallbackToLogcat || out == null) {
            Log.println(Log.INFO, "LocalMind.$tag", line.trim())
        }
    }

    fun flush() {
        try {
            writer?.flush()
        } catch (_: Throwable) {
        }
    }

    private fun now(): String =
        SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US).format(Date())
}
