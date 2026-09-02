#pragma once

// 平台无关的日志宏。
// Android 走 logcat，其它平台（含本机编译验证）走 stderr。

#if defined(__ANDROID__)
#include <android/log.h>

#define LM_LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "LocalMind", __VA_ARGS__)
#define LM_LOGW(...) __android_log_print(ANDROID_LOG_WARN,  "LocalMind", __VA_ARGS__)
#define LM_LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "LocalMind", __VA_ARGS__)
#define LM_LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "LocalMind", __VA_ARGS__)

#else
#include <cstdio>

#define LM_LOG_IMPL(level, ...)                        \
    do {                                               \
        std::fprintf(stderr, "[LocalMind][" level "] "); \
        std::fprintf(stderr, __VA_ARGS__);             \
        std::fprintf(stderr, "\n");                    \
    } while (0)

#define LM_LOGE(...) LM_LOG_IMPL("E", __VA_ARGS__)
#define LM_LOGW(...) LM_LOG_IMPL("W", __VA_ARGS__)
#define LM_LOGI(...) LM_LOG_IMPL("I", __VA_ARGS__)
#define LM_LOGD(...) LM_LOG_IMPL("D", __VA_ARGS__)

#endif
