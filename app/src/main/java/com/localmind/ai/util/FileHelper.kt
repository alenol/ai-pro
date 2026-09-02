package com.localmind.ai.util

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import java.io.File

// 把通过系统选择器拿到的 content:// Uri 复制进应用私有目录，
// 这样 native 层才能用 fopen 直接打开（llama.cpp 不认 content://）。
//
// 选择模型 / 视觉投影 / draft 文件时都需要这一步骤。
object FileHelper {

    fun fileNameOf(context: Context, uri: Uri): String {
        var name: String? = null
        runCatching {
            context.contentResolver.query(uri, null, null, null, null)?.use { c ->
                val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (idx >= 0 && c.moveToFirst()) name = c.getString(idx)
            }
        }
        return name ?: uri.lastPathSegment ?: "model.gguf"
    }

    // 复制到 getExternalFilesDir(subdir)/safeName。已存在同名则覆盖。
    // 返回绝对路径；失败返回 null。
    fun copyToAppFiles(
        context: Context,
        uri: Uri,
        subdir: String,
        overrideName: String? = null,
    ): String? = runCatching {
        val name = overrideName ?: fileNameOf(context, uri)
        val dir = File(context.getExternalFilesDir(null), subdir).apply { mkdirs() }
        val dst = File(dir, name)
        context.contentResolver.openInputStream(uri)?.use { input ->
            dst.outputStream().use { out -> input.copyTo(out) }
        }
        dst.absolutePath
    }.getOrNull()
}
