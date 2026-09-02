package com.localmind.ai

import android.app.Application
import com.localmind.ai.engine.AppServices
import com.localmind.ai.util.LogFile

// 应用入口。
//
// 所有重量级单例（推理运行时、向量库、HTTP 服务）都挂在 AppServices 上，
// 它在 Application.onCreate 里完成初始化（主要是加载 native 库、探测设备）。
//
// 这样设计的原因：本地模型是"设备级"资源，不该因为某个 Activity 被回收
// 就反复加载/卸载。多个 App 通过 AIDL 绑定时，也共用这一份实例。
class LocalMindApplication : Application() {

    lateinit var services: AppServices
        private set

    override fun onCreate() {
        super.onCreate()
        // 详细日志：构建时开 DEBUG_LOG 才启动（写 Download/localmodel/ 目录）
        LogFile.init(this, BuildConfig.DEBUG_LOG)
        try {
            services = AppServices(this)
            services.init()
        } catch (t: Throwable) {
            LogFile.e("Application", "onCreate init FAILED", t)
            throw t
        }
    }
}
