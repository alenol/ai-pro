package com.localmind.ai.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import com.localmind.ai.engine.ImagePayload

// 图片 -> native 推理需要的紧凑 RGB 负载。
// 注意：必须是 nx*ny*3 的紧凑数组，无行对齐填充，JNI 直接按 3 字节/像素读取。
object ImageUtils {

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

    // 从 Uri 解码（自动约束尺寸，避免大图 OOM）
    fun uriToPayload(context: Context, uri: Uri, maxSide: Int = 1024): ImagePayload? = runCatching {
        val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, opts)
        }
        val scale = (maxOf(opts.outWidth, opts.outHeight) / maxSide).coerceAtLeast(1)
        val load = BitmapFactory.Options().apply { inSampleSize = scale }
        val bmp = context.contentResolver.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, load)
        } ?: return null
        val payload = bitmapToPayload(bmp)
        bmp.recycle()
        payload
    }.getOrNull()
}
