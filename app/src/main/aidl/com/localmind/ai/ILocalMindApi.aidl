// LocalMind 对外 API（AIDL）。
//
// 其他 App 通过绑定服务使用本地模型，无需在自己的进程里加载一份模型。
// 模型是共享的：多个 App 调用会排队，不会各自占用一份内存。
//
// 用法见 docs/API.md。

package com.localmind.ai;

import com.localmind.ai.ILocalMindStreamCallback;

interface ILocalMindApi {

    /** 模型是否已加载并可用 */
    boolean isReady();

    /** 当前加载的模型标识 */
    String modelId();

    /**
     * 阻塞式生成。
     *
     * 注意：这个方法会一直阻塞到生成结束，请勿在调用方的主线程执行。
     * 需要实时输出时请用 generateStream。
     *
     * @param prompt      已渲染好的完整 prompt
     * @param paramsJson  采样参数 JSON，例如
     *                    {"temp":0.7,"topP":0.9,"nPredict":256}
     *                    传空串表示使用默认值
     * @return 生成的完整文本
     */
    String generate(String prompt, String paramsJson);

    /**
     * 流式生成。调用后立即返回，token 通过回调推送。
     *
     * @param cb 回调；传 null 时等价于 generate（但仍在后台执行，不返回结果）
     */
    oneway void generateStream(String prompt, String paramsJson, ILocalMindStreamCallback cb);

    /** 中断当前生成 */
    oneway void cancel();

    /**
     * 按模型自带的对话模板渲染消息。
     *
     * @param messagesJson [{"role":"user","content":"你好"}, ...]
     * @return 渲染后的 prompt；模型无模板时返回空串
     */
    String applyTemplate(String messagesJson);

    /**
     * 文本向量化。需要已加载 embedding 模型。
     * @return 向量；未加载时返回空数组
     */
    float[] embed(String text);

    /**
     * 本地知识库检索。
     * @return JSON 数组字符串：[{"title":"...","content":"..."}, ...]
     */
    String ragSearch(String query, int topK);

    /** 运行时信息：JSON，含 SoC / GPU / 内存 / OpenCL 可用性 */
    String runtimeInfo();

    /** 本地 HTTP 服务地址，未启动时返回空串 */
    String httpEndpoint();
}
