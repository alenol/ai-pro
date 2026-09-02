package com.localmind.ai.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.localmind.ai.engine.ImagePayload
import com.localmind.ai.engine.PerfPreset
import com.localmind.ai.rag.DocRow

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    vm: MainViewModel,
    pendingImages: List<ImagePayload>,
    onPickModel: () -> Unit,
    onPickMmproj: () -> Unit,
    onPickDraft: () -> Unit,
    onPickImage: () -> Unit,
    onPickDoc: () -> Unit,
    onSend: (String) -> Unit,
    onClearImages: () -> Unit,
) {
    val tabs = listOf("模型", "对话", "知识库", "API", "设置")
    var tab by remember { mutableStateOf(0) }

    Scaffold(
        topBar = { TopAppBar(title = { Text("LocalMind · 本地 AI") }) }
    ) { pad ->
        Column(modifier = Modifier.fillMaxSize().padding(pad)) {
            TabRow(selectedTabIndex = tab) {
                tabs.forEachIndexed { i, t -> Tab(selected = tab == i, onClick = { tab = i }, text = { Text(t) }) }
            }
            when (tab) {
                0 -> ModelScreen(vm, onPickModel, onPickMmproj, onPickDraft)
                1 -> ChatScreen(vm, pendingImages, onPickImage, onClearImages, onSend)
                2 -> KnowledgeScreen(vm, onPickDoc)
                3 -> ApiScreen(vm)
                4 -> SettingsScreen(vm)
            }
        }
    }
}

// =====================================================================
// 模型
// =====================================================================
@Composable
private fun ModelScreen(
    vm: MainViewModel,
    onPickModel: () -> Unit,
    onPickMmproj: () -> Unit,
    onPickDraft: () -> Unit,
) {
    val scroll = rememberScrollState()
    Column(modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(scroll)) {
        Text("主模型（GGUF）", style = MaterialTheme.typography.titleMedium)
        PathRow(vm.modelPath.ifEmpty { "未选择" }) { onPickModel() }

        Text("视觉投影 mmproj（多模态可选）", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 12.dp))
        PathRow(vm.mmprojPath.ifEmpty { "未选择（纯文本模型可留空）" }) { onPickMmproj() }

        Text("Draft 模型（自投机解码可选）", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 12.dp))
        PathRow(vm.draftPath.ifEmpty { "未选择（留空则用 MTP）" }) { onPickDraft() }

        Text("性能档位", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 12.dp))
        PerfPreset.values().forEach { p ->
            Row(verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.fillMaxWidth().clickable { vm.preset = p }) {
                RadioButton(selected = vm.preset == p, onClick = { vm.preset = p })
                Column(modifier = Modifier.padding(start = 4.dp)) {
                    Text(p.label)
                    Text(p.desc, style = MaterialTheme.typography.labelSmall)
                }
            }
        }

        Row(modifier = Modifier.padding(top = 16.dp)) {
            Button(onClick = { vm.loadModel() }, enabled = !vm.busy) { Text("加载") }
            Spacer(Modifier.width(8.dp))
            OutlinedButton(onClick = { vm.unload() }, enabled = vm.modelLoaded) { Text("卸载") }
        }

        Card(modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(vm.status)
                vm.modelInfo?.let {
                    Spacer(Modifier.height(6.dp))
                    Text("上下文: ${it.nCtx}  维度: ${it.nEmbd}  视觉: ${if (it.vision) "支持" else "不支持"}")
                    Text("后端: ${it.backend}")
                }
            }
        }
    }
}

@Composable
private fun PathRow(path: String, onClick: () -> Unit) {
    OutlinedButton(onClick = onClick, modifier = Modifier.fillMaxWidth().padding(top = 4.dp)) {
        Text(path, maxLines = 1, modifier = Modifier.fillMaxWidth())
    }
}

// =====================================================================
// 对话
// =====================================================================
@Composable
private fun ChatScreen(
    vm: MainViewModel,
    pendingImages: List<ImagePayload>,
    onPickImage: () -> Unit,
    onClearImages: () -> Unit,
    onSend: (String) -> Unit,
) {
    var input by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    LaunchedEffect(vm.messages.size) {
        if (vm.messages.isNotEmpty()) listState.scrollToItem(vm.messages.lastIndex)
    }

    Column(modifier = Modifier.fillMaxSize()) {
        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth().padding(8.dp)) {
            Text("使用知识库", style = MaterialTheme.typography.labelMedium)
            Switch(checked = vm.useRag, onCheckedChange = { vm.useRag = it })
            if (!vm.hasKnowledge) Text("（知识库为空）", style = MaterialTheme.typography.labelSmall)
            Spacer(Modifier.weight(1f))
            if (vm.generating)
                OutlinedButton(onClick = { vm.cancel() }) { Text("停止") }
        }

        LazyColumn(
            state = listState,
            modifier = Modifier.fillMaxWidth().weight(1f).padding(horizontal = 12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            items(vm.messages, key = { it.id }) { msg -> MessageBubble(msg) }
        }

        if (pendingImages.isNotEmpty()) {
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(horizontal = 12.dp)) {
                Text("已附 ${pendingImages.size} 张图", style = MaterialTheme.typography.labelSmall)
                Spacer(Modifier.width(8.dp))
                TextButton_(text = "清除") { onClearImages() }
            }
        }

        Row(modifier = Modifier.fillMaxWidth().padding(8.dp), verticalAlignment = Alignment.Bottom) {
            Button(onClick = onPickImage, enabled = !vm.generating) { Text("图片") }
            Spacer(Modifier.width(8.dp))
            OutlinedTextField(
                value = input, onValueChange = { input = it },
                modifier = Modifier.weight(1f), placeholder = { Text("输入消息…") },
                maxLines = 4,
            )
            Spacer(Modifier.width(8.dp))
            Button(onClick = {
                if (input.isNotBlank() && !vm.generating) { onSend(input); input = "" }
            }, enabled = !vm.generating) { Text("发送") }
        }
        if (vm.genHint.isNotEmpty())
            Text(vm.genHint, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp))
    }
}

@Composable
private fun MessageBubble(msg: com.localmind.ai.ui.UiMessage) {
    val isUser = msg.role == "user"
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start) {
        Card(
            colors = CardDefaults.cardColors(
                containerColor = if (isUser) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.surfaceVariant,
            ),
            modifier = Modifier.fillMaxWidth(0.9f),
        ) {
            Column(modifier = Modifier.padding(10.dp)) {
                if (msg.images.isNotEmpty())
                    Text("🖼 ${msg.images.size} 张图片", style = MaterialTheme.typography.labelSmall)
                Text(msg.text.ifEmpty { if (msg.isStreaming) "…" else "" },
                    color = if (isUser) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurface)
                if (msg.error != null)
                    Text("⚠ ${msg.error}", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
            }
        }
    }
}

// =====================================================================
// 知识库
// =====================================================================
@Composable
private fun KnowledgeScreen(vm: MainViewModel, onPickDoc: () -> Unit) {
    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Text("本地知识库（混合检索：BM25 + 向量）", style = MaterialTheme.typography.titleMedium)
        if (vm.ragStatus.isNotEmpty()) Text(vm.ragStatus, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 4.dp))

        Row(modifier = Modifier.padding(vertical = 12.dp)) {
            Button(onClick = onPickDoc, enabled = !vm.ragBusy) { Text("导入文档") }
            Spacer(Modifier.width(8.dp))
            OutlinedButton(onClick = { vm.refreshDocs() }) { Text("刷新") }
            Spacer(Modifier.width(8.dp))
            OutlinedButton(onClick = { vm.clearKnowledge() }) { Text("清空") }
        }

        Text("提示：向量检索需先加载 embedding 模型（在「设置」配置路径）。未加载时仅使用 BM25 关键词索引。",
            style = MaterialTheme.typography.labelSmall)

        Spacer(Modifier.height(8.dp))
        HorizontalDivider()
        LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
            items(vm.docs, key = { it.id }) { doc -> DocItem(doc) { vm.deleteDoc(doc.id) } }
        }
    }
}

@Composable
private fun DocItem(doc: DocRow, onDelete: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
        Row(modifier = Modifier.fillMaxWidth().padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(modifier = Modifier.weight(1f)) {
                Text(doc.title)
                Text("${doc.nChunks} 段 · ${doc.mime} · ${java.text.SimpleDateFormat("MM-dd HH:mm").format(java.util.Date(doc.createdAt))}",
                    style = MaterialTheme.typography.labelSmall)
            }
            TextButton_(text = "删除") { onDelete() }
        }
    }
}

// =====================================================================
// 本地 API
// =====================================================================
@Composable
private fun ApiScreen(vm: MainViewModel) {
    val scroll = rememberScrollState()
    Column(modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(scroll)) {
        Text("本地 API（OpenAI 兼容）", style = MaterialTheme.typography.titleMedium)
        Text("其他 App 或脚本可直接调用，无需自己加载模型。默认仅本机 127.0.0.1。",
            style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 4.dp))

        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 12.dp)) {
            Text("启用 HTTP 服务", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            Switch(checked = vm.apiEnabled, onCheckedChange = { vm.toggleApi() })
        }

        Text("端口", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 8.dp))
        OutlinedTextField(value = vm.apiPort, onValueChange = { vm.apiPort = it }, modifier = Modifier.fillMaxWidth())

        Text("API Key（留空不校验）", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 8.dp))
        OutlinedTextField(value = vm.apiKey, onValueChange = { vm.apiKey = it }, modifier = Modifier.fillMaxWidth())

        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 8.dp)) {
            Text("允许局域网访问（0.0.0.0）", style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.weight(1f))
            Switch(checked = vm.apiLan, onCheckedChange = { vm.apiLan = it })
        }

        if (vm.apiEndpoint.isNotEmpty()) {
            Card(modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text("服务地址")
                    Text(vm.apiEndpoint, style = MaterialTheme.typography.titleSmall)
                    Spacer(Modifier.height(6.dp))
                    Text("端点：/v1/chat/completions · /v1/embeddings · /v1/models · /v1/rag/search")
                }
            }
        }

        Spacer(Modifier.height(12.dp))
        Text("AIDL 接口", style = MaterialTheme.typography.titleMedium)
        Text("其他 App 可绑定 com.localmind.ai.LocalMindApiService 使用 ILocalMindApi，模型与本 App 共享，不重复占内存。",
            style = MaterialTheme.typography.labelSmall)
    }
}

// =====================================================================
// 设置
// =====================================================================
@Composable
private fun SettingsScreen(vm: MainViewModel) {
    val scroll = rememberScrollState()
    val p = vm.profile
    Column(modifier = Modifier.fillMaxSize().padding(16.dp).verticalScroll(scroll)) {
        Text("设备画像", style = MaterialTheme.typography.titleMedium)
        Card(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("SoC: ${p.socModel}")
                Text("GPU: ${p.gpuName}")
                Text("CPU 核心: ${p.nCores}")
                Text("内存: ${p.totalRamMb} MB")
                Text("OpenCL: ${if (p.openclAvailable) "可用（GPU 加速）" else "不可用（将纯 CPU）"}")
            }
        }

        Text("骁龙专属调优建议", style = MaterialTheme.typography.titleMedium, modifier = Modifier.padding(top = 16.dp))
        vm.tuningTips().forEach {
            Text("· $it", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 4.dp))
        }

        Spacer(Modifier.height(16.dp))
        Text("Embedding 模型路径（知识库向量检索用）", style = MaterialTheme.typography.titleMedium)
        Text("把 Qwen3-Embedding 等 embedding GGUF 放到本机，并在加载后导入文档即可获得语义检索能力。",
            style = MaterialTheme.typography.labelSmall)
    }
}

// 轻量文字按钮，避免重复引入 TextButton 的导入噪音
@Composable
private fun TextButton_(text: String, onClick: () -> Unit) {
    Box(modifier = Modifier.clickable { onClick() }.padding(8.dp)) {
        Text(text, color = MaterialTheme.colorScheme.primary)
    }
}
