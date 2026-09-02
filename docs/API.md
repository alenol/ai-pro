# 本地 API 用法

LocalMind 对外暴露两套 API，模型对调用方是**共享**的（不重复加载权重）：

1. **OpenAI 兼容 HTTP**：默认监听 `127.0.0.1:8080`，任何 OpenAI SDK 都能直接连。
2. **AIDL**：供其他安卓 APK 在进程内跨进程调用，无需自己加载模型。

---

## 一、HTTP（OpenAI 兼容）

### 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查 |
| GET/POST | `/v1/models` | 当前模型列表 |
| POST | `/v1/chat/completions` | 对话（支持 `stream:true` SSE） |
| POST | `/v1/completions` | 续写 |
| POST | `/v1/embeddings` | 文本向量化（需 embedding 模型） |
| POST | `/v1/rag/search` | 本地知识库检索（扩展端点，非 OpenAI 标准） |

> 默认只监听回环地址。`API` 页勾选「允许局域网访问」会监听 `0.0.0.0`（建议同时设置 API Key）。

### 示例：curl

```bash
# 对话
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"用一句话解释 MTP 投机解码"}],"max_tokens":256}'

# 流式
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"你好"}],"stream":true}'

# 向量化
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input":"骁龙8至尊版"}'

# 知识库检索
curl http://127.0.0.1:8080/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{"query":"如何省电","top_k":3}'
```

### 示例：Python（OpenAI 客户端）

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="not-needed")

resp = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

---

## 二、AIDL（跨 App 调用）

- **服务类**：`com.localmind.ai.LocalMindApiService`
- **接口包**：`com.localmind.ai`（AIDL 文件在 `app/src/main/aidl/com/localmind/ai/`）

### 接口方法（`ILocalMindApi`）

| 方法 | 说明 |
|---|---|
| `boolean isReady()` | 模型是否已加载 |
| `String modelId()` | 当前模型路径/标识 |
| `String generate(String prompt, String paramsJson)` | 阻塞式生成，返回完整文本（会一直阻塞到结束，勿在主线程调用） |
| `oneway void generateStream(String prompt, String paramsJson, ILocalMindStreamCallback cb)` | 流式生成，token 经回调推送 |
| `oneway void cancel()` | 中断当前生成 |
| `String applyTemplate(String messagesJson)` | 按模型自带模板渲染对话（输入 `[{"role","content"}]` 的 JSON） |
| `float[] embed(String text)` | 文本向量化（需 embedding 模型） |
| `String ragSearch(String query, int topK)` | 知识库检索，返回 `[{"title","content"}]` JSON |
| `String runtimeInfo()` | 运行时信息（SoC/GPU/内存/OpenCL） |
| `String httpEndpoint()` | 本地 HTTP 服务地址 |

`paramsJson` 示例：`{"temp":0.7,"topP":0.9,"nPredict":256}`，空串用默认。

### 客户端示例（Kotlin）

把 `ILocalMindApi.aidl`、`ILocalMindStreamCallback.aidl` 复制到你的工程的 `aidl/com/localmind/ai/` 下。

```kotlin
class MyClient : ServiceConnection {
    private var api: ILocalMindApi? = null

    override fun onServiceConnected(name: ComponentName, binder: IBinder) {
        api = ILocalMindApi.Stub.asInterface(binder)
        if (api?.isReady == true) {
            val text = api!!.generate("你好", """{"nPredict":128}""")
            Log.d("LM", text)
        }
    }
    override fun onServiceDisconnected(name: ComponentName) { api = null }
}

// 绑定（显式组件，因为服务未声明 action）
val intent = Intent().apply {
    component = ComponentName("com.localmind.ai", "com.localmind.ai.LocalMindApiService")
}
bindService(intent, MyClient(), Context.BIND_AUTO_CREATE)
```

### 流式回调示例

```kotlin
val cb = object : ILocalMindStreamCallback.Stub() {
    override fun onToken(text: String) { print(text) }
    override fun onDone(statsJson: String) { println("\n[完成] $statsJson") }
}
api?.generateStream("讲个笑话", "{}", cb)
```
