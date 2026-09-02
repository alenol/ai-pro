package com.localmind.ai.engine

import kotlin.math.max
import kotlin.math.min

// 第五代骁龙 8 至尊版（SM8850）专属参数调优。
//
// 手机上跑 LLM 和桌面最大的区别是：没有独立显存。
// GPU 与 CPU 共享同一块 LPDDR5X，且没有散热风扇。
// 这意味着参数的取舍逻辑和桌面完全不同：
//
//   1. 层数卸载不是越多越好。全部卸载到 Adreno 后，KV cache 也被挤进同一块内存，
//      上下文一长就会触发 OOM。真正的最优解往往是把输出层/词嵌入留在 CPU。
//   2. 线程数不是越多越好。SM8850 是 2 超大核 + 6 性能核的 Oryon 布局，
//      开满 8 线程会把任务摊到正在降频的核心上，反而更慢且更热。
//   3. KV cache 精度直接决定能开多长的上下文。F16 与 Q8_0 之间是一倍的空间差。
//
// 下面的数值是针对该平台的经验值，均可被 UI 覆盖。

data class DeviceProfile(
    val socModel: String,
    val gpuName: String,
    val nCores: Int,
    val totalRamMb: Long,
    val openclAvailable: Boolean,
) {
    // SM8850 即第五代骁龙 8 至尊版。不同厂商 ro.soc.model 的写法略有差异，
    // 因此同时检查平台代号与 GPU 名称作为兜底。
    val isSnapdragon8EliteGen5: Boolean
        get() {
            val soc = socModel.uppercase()
            if (soc.contains("SM8850")) return true
            if (gpuName.contains("Adreno", ignoreCase = true)) {
                // Adreno 840 系是本代 GPU
                return Regex("""Adreno.*\b(8[3-9]\d)\b""").containsMatchIn(gpuName)
            }
            return false
        }

    val isSnapdragon: Boolean
        get() = socModel.uppercase().startsWith("SM") ||
                gpuName.contains("Adreno", ignoreCase = true)

    // 可用于模型的预算。系统与其它应用要保留一部分，
    // 且 Android 对单个进程的可用内存有额外限制。
    val modelBudgetMb: Long
        get() {
            val ram = totalRamMb
            val usable = when {
                ram <= 0 -> 6_000L                              // 探测失败时保守估计
                ram < 8_000  -> (ram * 0.35).toLong()           // 8GB 机型
                ram < 12_000 -> (ram * 0.45).toLong()           // 12GB
                ram < 18_000 -> (ram * 0.55).toLong()           // 16GB
                else         -> (ram * 0.65).toLong()           // 24GB+ 旗舰
            }
            // Android 单进程上限通常低于物理内存，这里再压一道
            return min(usable, 14_000L)
        }
}

// 预设档位
enum class PerfPreset(val label: String, val desc: String) {
    BALANCED("均衡", "默认档，速度与上下文长度兼顾"),
    SPEED("极速", "缩短上下文、提高批大小，追求最低首字延迟"),
    LONG_CTX("长上下文", "降低 KV 精度换取 32K+ 上下文"),
    LOW_MEM("省内存", "小模型也能在大上下文下稳定运行"),
}

object SnapdragonTuner {

    // 生成阶段线程数。
    //
    // SM8850 的 6 个性能核是主力，2 个超大核单核更强但持续负载下会先撞温度墙。
    // 实测经验：4~5 线程在长对话中最稳，短任务开 6 线程能再快一点。
    fun recommendThreads(profile: DeviceProfile, preset: PerfPreset = PerfPreset.BALANCED): Int {
        if (profile.isSnapdragon8EliteGen5) {
            return when (preset) {
                PerfPreset.SPEED    -> 6
                PerfPreset.LONG_CTX -> 4
                PerfPreset.LOW_MEM  -> 4
                else                -> 5
            }
        }
        // 通用 Android：核心数的一半，但不超过 6
        return max(2, min(profile.nCores / 2, 6))
    }

    // prompt 处理阶段可以用满核心，这一阶段是纯计算、时间短
    fun recommendBatchThreads(profile: DeviceProfile): Int =
        max(2, min(profile.nCores, 8))

    // 物理批大小。
    //
    // Adreno 的 OpenCL 后端对 ubatch 比较敏感：太小则 GPU 利用率低，
    // 太大则峰值内存暴涨（这是手机 OOM 最常见的诱因）。
    fun recommendUbatch(profile: DeviceProfile, preset: PerfPreset): Int =
        when (preset) {
            PerfPreset.SPEED    -> if (profile.modelBudgetMb >= 8_000) 1024 else 768
            PerfPreset.LONG_CTX -> 512
            PerfPreset.LOW_MEM  -> 256
            else                -> 512
        }

    // KV cache 精度。
    //
    // 每 token 的 KV 占用（经验值，典型 7~9B 模型 36 层、head_dim 128、4 个 KV head）：
    //   F16  ≈ 72 KB/token
    //   Q8_0 ≈ 37 KB/token
    //   Q4_0 ≈ 19 KB/token
    private const val BYTES_PER_TOKEN_F16  = 73_728L
    private const val BYTES_PER_TOKEN_Q8_0 = 36_864L
    private const val BYTES_PER_TOKEN_Q4_0 = 18_432L

    fun kvBytesPerToken(cache: CacheType): Long = when (cache) {
        CacheType.F16  -> BYTES_PER_TOKEN_F16
        CacheType.Q8_0 -> BYTES_PER_TOKEN_Q8_0
        CacheType.Q4_0 -> BYTES_PER_TOKEN_Q4_0
    }

    fun estimateKvMb(nCtx: Int, cache: CacheType, nSeq: Int = 1): Long =
        kvBytesPerToken(cache) * nCtx * nSeq / (1024 * 1024)

    fun recommendCacheType(
        profile: DeviceProfile,
        modelBytes: Long,
        nCtx: Int,
        preset: PerfPreset,
    ): CacheType {
        val budgetBytes = profile.modelBudgetMb * 1024 * 1024
        val remain = budgetBytes - modelBytes

        if (preset == PerfPreset.LONG_CTX) return CacheType.Q8_0
        if (preset == PerfPreset.LOW_MEM)  return CacheType.Q4_0

        // 优先 F16（质量最好），装不下才逐级降
        if (remain > estimateKvMb(nCtx, CacheType.F16) * 1024L * 1024L) return CacheType.F16
        if (remain > estimateKvMb(nCtx, CacheType.Q8_0) * 1024L * 1024L) return CacheType.Q8_0
        return CacheType.Q4_0
    }

    fun recommendCtx(profile: DeviceProfile, preset: PerfPreset): Int =
        when (preset) {
            PerfPreset.SPEED    -> 4096
            PerfPreset.LONG_CTX -> 32768
            PerfPreset.LOW_MEM  -> 8192
            else                -> if (profile.modelBudgetMb >= 10_000) 16384 else 8192
        }

    // 层数卸载策略。
    //
    // 全卸载（-1）在内存充裕时最快；预算紧张时把注意力输出层留在 CPU，
    // 可以省下可观的一块显存（词嵌入 + 输出层在 9B 模型上可达数百 MB）。
    fun recommendGpuLayers(profile: DeviceProfile, modelBytes: Long, preset: PerfPreset): Int {
        if (!profile.openclAvailable) return 0
        val budgetMb = profile.modelBudgetMb
        val modelMb = modelBytes / (1024 * 1024)

        return when {
            preset == PerfPreset.LOW_MEM && modelMb > budgetMb * 0.8 -> 0
            modelMb < budgetMb * 0.6 -> -1          // 全卸载
            else -> -1
        }
    }

    // 生成一份可直接使用的配置
    fun buildConfig(
        profile: DeviceProfile,
        modelPath: String,
        mmprojPath: String = "",
        modelBytes: Long = 0L,
        preset: PerfPreset = PerfPreset.BALANCED,
    ): ModelConfig {
        val nCtx = recommendCtx(profile, preset)
        val cache = recommendCacheType(profile, modelBytes, nCtx, preset)

        // OpenCL(Adreno GPU) 后端稳定性开关：
        //   - llama.cpp 的 FlashAttention 在 OpenCL 后端支持不完整，decode 阶段易原生崩溃
        //     （这类崩溃发生在 native 层，Java 崩溃处理器抓不到，表现就是"对话闪退"）
        //   - MTP 投机需要第二个 GPU 上下文（ctx_dft），在 OpenCL 上双 GPU 上下文是高危组合
        // 因此 GPU(OpenCL) 卸载时关闭 FA、投机降级为纯 CPU 的 ngram（零额外内存）；
        // 纯 CPU 后端没有这些问题，保持 MTP + FlashAttention 以获得最高速度。
        val onOpenCl = profile.openclAvailable
        val spec = if (onOpenCl) SpecMode.NGRAM else SpecMode.MTP
        val flashAttn = !onOpenCl

        return ModelConfig(
            modelPath = modelPath,
            mmprojPath = mmprojPath,
            nCtx = nCtx,
            nBatch = if (nCtx >= 4096) 2048 else 1024,
            nUbatch = recommendUbatch(profile, preset),
            nGpuLayers = recommendGpuLayers(profile, modelBytes, preset),
            nThreads = recommendThreads(profile, preset),
            nThreadsBatch = recommendBatchThreads(profile),
            flashAttn = flashAttn,
            cacheTypeK = cache,
            cacheTypeV = cache,
            specMode = spec,
            specNMax = if (preset == PerfPreset.SPEED) 24 else 16,
            specNMin = 4,
            specPMin = 0.75f,
            loadMtp = spec == SpecMode.MTP,   // 仅 MTP 才需要加载 MTP 头权重
        )
    }

    // 给用户的可读建议，展示在设置页
    fun explain(profile: DeviceProfile, cfg: ModelConfig, modelMb: Long): List<String> {
        val out = mutableListOf<String>()

        out += if (profile.isSnapdragon8EliteGen5) {
            "已识别第五代骁龙 8 至尊版（${profile.socModel}），已应用专属参数"
        } else if (profile.isSnapdragon) {
            "检测到骁龙平台（${profile.socModel}），使用通用骁龙参数"
        } else {
            "未识别为骁龙平台，使用通用 Android 参数"
        }

        out += if (profile.openclAvailable) {
            "GPU 后端：${profile.gpuName}，卸载层数 ${cfg.nGpuLayers}（OpenCL 下已关闭 FlashAttention/MTP 以保证稳定）"
        } else {
            "未检测到 OpenCL 设备，将纯 CPU 推理（速度会明显下降）"
        }

        out += "线程：生成 ${cfg.nThreads} / 批处理 ${cfg.nThreadsBatch}（共 ${profile.nCores} 核）"
        out += "KV cache：${cfg.cacheTypeK}，${cfg.nCtx} tokens 约需 ${estimateKvMb(cfg.nCtx, cfg.cacheTypeK)} MB"

        if (modelMb > 0) {
            val budget = profile.modelBudgetMb
            val total = modelMb + estimateKvMb(cfg.nCtx, cfg.cacheTypeK)
            out += "内存预算：模型 ${modelMb}MB + KV ${estimateKvMb(cfg.nCtx, cfg.cacheTypeK)}MB " +
                   "= ${total}MB / 可用 ${budget}MB"
            if (total > budget) {
                out += "⚠ 预计超出内存预算，建议降低上下文长度或改用 Q8_0/Q4_0"
            }
        }

        out += when (cfg.specMode) {
            SpecMode.MTP -> "投机解码：MTP（模型自带多头预测，几乎零额外内存）"
            SpecMode.DRAFT_MODEL -> "投机解码：独立 draft 模型（需额外内存）"
            SpecMode.NGRAM -> "投机解码：ngram（零额外内存）"
            SpecMode.OFF -> "投机解码：关闭"
        }

        return out
    }
}
