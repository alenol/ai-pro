package com.localmind.ai.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.mutableStateListOf
import com.localmind.ai.engine.ImagePayload
import com.localmind.ai.util.FileHelper
import com.localmind.ai.util.ImageUtils

class MainActivity : ComponentActivity() {

    private val vm: MainViewModel by viewModels()

    // 聊天待发送图片（已转成 RGB 负载）
    private val pendingImages = mutableStateListOf<ImagePayload>()

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        val path = FileHelper.copyToAppFiles(this, uri, "models", "main.gguf")
        if (path != null) vm.modelPath = path else vm.status = "模型复制失败"
    }
    private val pickMmproj = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        val path = FileHelper.copyToAppFiles(this, uri, "models", "mmproj.gguf")
        if (path != null) vm.mmprojPath = path else vm.status = "视觉投影文件复制失败"
    }
    private val pickDraft = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        val path = FileHelper.copyToAppFiles(this, uri, "models", "draft.gguf")
        if (path != null) vm.draftPath = path else vm.status = "draft 模型复制失败"
    }
    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri ?: return@registerForActivityResult
        ImageUtils.uriToPayload(this, uri)?.let { pendingImages.add(it) }
    }
    private val pickDoc = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri ?: return@registerForActivityResult
        vm.importDocument(uri)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            LocalMindTheme {
                MainScreen(
                    vm = vm,
                    pendingImages = pendingImages,
                    onPickModel = { pickModel.launch(arrayOf("*/*")) },
                    onPickMmproj = { pickMmproj.launch(arrayOf("*/*")) },
                    onPickDraft = { pickDraft.launch(arrayOf("*/*")) },
                    onPickImage = { pickImage.launch("image/*") },
                    onPickDoc = { pickDoc.launch(arrayOf("*/*")) },
                    onSend = { text ->
                        vm.send(text, pendingImages.toList())
                        pendingImages.clear()
                    },
                    onClearImages = { pendingImages.clear() },
                )
            }
        }
    }
}
