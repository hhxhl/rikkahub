package me.rerere.locallm.litert

import android.content.Context
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.BenchmarkInfo
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.benchmark
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import me.rerere.locallm.LocalRuntime
import me.rerere.locallm.LocalRuntimePreferences

/** Standalone LiteRT-LM 0.13.1 benchmark; never uses the chat Engine/Conversation. */
class LiteRtBenchmarkRunner(
    private val context: Context,
    private val runtime: LiteRtRuntime,
    private val preferences: LocalRuntimePreferences,
) {
    data class Run(val index: Int, val metrics: BenchmarkInfo)

    data class Result(
        val modelName: String,
        val accelerator: String,
        val prefillTokens: Int,
        val decodeTokens: Int,
        val startedAtMs: Long,
        val finishedAtMs: Long,
        val runs: List<Run>,
    ) {
        val prefillTps: Double get() = runs.map { it.metrics.lastPrefillTokensPerSecond }.average()
        val decodeTps: Double get() = runs.map { it.metrics.lastDecodeTokensPerSecond }.average()
        val ttftSeconds: Double get() = runs.map { it.metrics.timeToFirstTokenInSecond }.average()
        val firstInitMs: Double get() = runs.first().metrics.initTimeInSecond * 1000.0
        val steadyInitMs: Double? get() = runs.drop(1).takeIf { it.isNotEmpty() }
            ?.map { it.metrics.initTimeInSecond * 1000.0 }?.average()

        /** One-row CSV with the Box 3.5.7 summary field names/order for comparisons. */
        fun toCsv(appVersion: String): String {
            val header = listOf(
                "start time (ms)", "end time (ms)", "model name", "accelerator",
                "prefill tokens count", "decode tokens count", "runs count", "app version",
                "prefill speed (tokens/sec)", "decode speed (tokens/sec)",
                "time to first token (sec)", "first init time (ms)", "steady init time (ms)",
            ).joinToString(",")
            val row = listOf(
                startedAtMs.toString(), finishedAtMs.toString(), csvCell(modelName),
                csvCell(accelerator), prefillTokens.toString(), decodeTokens.toString(),
                runs.size.toString(), csvCell(appVersion),
                prefillTps.toString(), decodeTps.toString(), ttftSeconds.toString(),
                firstInitMs.toString(), steadyInitMs?.toString().orEmpty(),
            ).joinToString(",")
            return "$header\n$row\n"
        }
    }

    @OptIn(ExperimentalApi::class)
    suspend fun run(
        modelId: String,
        accelerator: String,
        prefillTokens: Int = 256,
        decodeTokens: Int = 256,
        runCount: Int = 3,
        onProgress: suspend (Int) -> Unit = {},
    ): Result = withContext(Dispatchers.IO) {
        require(prefillTokens in 1..8192) { "Prefill tokens must be between 1 and 8192" }
        require(decodeTokens in 1..4096) { "Decode tokens must be between 1 and 4096" }
        require(runCount in 1..10) { "Run count must be between 1 and 10" }
        require(accelerator in setOf("CPU", "GPU")) { "Unsupported accelerator" }

        val modelPath = preferences.installedModels(LocalRuntime.LiteRT)[modelId]
            ?: error("Model is not installed: $modelId")
        val modelFile = File(modelPath)
        require(modelFile.isFile) { "Model file does not exist: $modelId" }
        // The native benchmark does not know our EngineConfig clamp. In particular,
        // requesting more than a baked ekvNNNN cache may crash JNI instead of throwing.
        val ceiling = LiteRtModelFile.resolveContextCeiling(
            modelFile, modelId, LiteRtModelDefaults.forModelFile(modelId),
        )
        if (ceiling != null) {
            require(prefillTokens.toLong() + decodeTokens + 32L <= ceiling) {
                "Prefill + decode + 32 must fit the model's $ceiling-token KV cache"
            }
        }

        runtime.withExclusiveBenchmark(modelPath, accelerator) {
            // All runs use the same isolated cache folder, allowing cold / steady
            // initialization measurements without modifying the chat cache.
            val cache = File(context.cacheDir, "litert-benchmark-${System.nanoTime()}")
            check(cache.mkdirs()) { "Cannot create benchmark cache directory" }
            try {
                val backend = if (accelerator == "GPU") Backend.GPU() else Backend.CPU()
                val startedAtMs = System.currentTimeMillis()
                val results = mutableListOf<Run>()
                for (index in 0 until runCount) {
                    currentCoroutineContext().ensureActive()
                    // This call creates and releases native benchmark resources itself.
                    val info = benchmark(
                        modelPath = modelPath,
                        backend = backend,
                        prefillTokens = prefillTokens,
                        decodeTokens = decodeTokens,
                        cacheDir = cache.absolutePath,
                    )
                    results.add(Run(index + 1, info))
                    onProgress(index + 1)
                }
                Result(
                    modelName = modelId,
                    accelerator = accelerator,
                    prefillTokens = prefillTokens,
                    decodeTokens = decodeTokens,
                    startedAtMs = startedAtMs,
                    finishedAtMs = System.currentTimeMillis(),
                    runs = results,
                )
            } finally {
                // Do not remove the model's normal compiled cache or local files.
                cache.deleteRecursively()
            }
        }
    }
}

/** Only model/version cells need CSV escaping; numeric columns are generated internally. */
private fun csvCell(s: String): String {
    // Prevent spreadsheet formula interpretation of filenames imported by the user.
    val neutral = if (s.startsWith('=') || s.startsWith('+') ||
        s.startsWith('-') || s.startsWith('@')) "'$s" else s
    return "\"${neutral.replace("\"", "\"\"")}\""
}
