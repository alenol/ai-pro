package com.localmind.ai;

// 流式生成的回调。
// onToken 在推理线程被调用，实现里应尽快返回，不要做耗时操作。
interface ILocalMindStreamCallback {

    /** 增量文本。返回 false 可中止生成（AIDL oneway 下返回值不可用，见 generateStream 说明） */
    void onToken(String text);

    /**
     * 生成结束。
     * @param statsJson {"stopReason":"eos","nGenTokens":42,"genTps":18.3,
     *                   "nDrafted":120,"nAccepted":88,"acceptRate":0.733}
     */
    void onDone(String statsJson);
}
