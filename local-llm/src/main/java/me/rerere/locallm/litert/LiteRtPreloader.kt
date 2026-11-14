package me.rerere.locallm.litert

import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import me.rerere.locallm.LocalRuntime
import me.rerere.locallm.LocalRuntimePreferences
import java.io.File

/** Lightweight model-selection adapter. Engine-only preload; no chat history or tools here. */
class LiteRtPreloader(
    private val prefs: LocalRuntimePreferences,
    private val runtime: LiteRtRuntime,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val requests = Channel<String?>(Channel.CONFLATED)
    @Volatile private var selectedModel: String? = null

    init {
        scope.launch {
            for (requested in requests) {
                // Debounce quick model picker navigation without cancelling a native init.
                delay(400)
                if (requested != selectedModel) continue
                if (requested == null) continue
                try {
                    // If an earlier request is between ensureLoaded and streamTurns,
                    // retry rather than evicting its Engine. Active inference itself is
                    // serialized by the runtime mutex.
                    for (attempt in 0 until 30) {
                        if (requested != selectedModel) break
                        if (preload(requested) != LiteRtRuntime.PreloadResult.DEFERRED) break
                        delay(1000)
                    }
                    // The user may have switched to another provider while loading.
                    if (requested != selectedModel) {
                        val oldPath = prefs.installedModels(LocalRuntime.LiteRT)[requested]
                        runtime.discardPreloadedEngine(oldPath)
                    }
                } catch (t: Throwable) {
                    Log.w("LiteRtPreloader", "Engine preload skipped for $requested", t)
                }
            }
        }
    }

    /** Fire-and-forget; normal Provider.ensureLoaded remains the fallback on any failure. */
    fun onModelSelected(modelFile: String?) {
        if (selectedModel == modelFile) return
        selectedModel = modelFile
        requests.trySend(modelFile)
    }

    private suspend fun preload(modelFile: String): LiteRtRuntime.PreloadResult {
        val path = prefs.installedModels(LocalRuntime.LiteRT)[modelFile]
            ?: return LiteRtRuntime.PreloadResult.FAILED
        if (!File(path).isFile) return LiteRtRuntime.PreloadResult.FAILED
        val config = LiteRtModelDefaults.forModelFile(modelFile)
        val contextCeiling = LiteRtModelFile.resolveContextCeiling(
            File(path), modelFile, config,
        )
        val requestedContext = prefs.maxNumTokensOverride(LocalRuntime.LiteRT) ?: config.maxTokens
        val tokens = LiteRtProvider.engineContextTokens(requestedContext, contextCeiling)
        val imageSupport = config.supportsImage && config.visionAccelerator != null &&
            !prefs.isVisionUnavailable(LocalRuntime.LiteRT, modelFile)
        return runtime.preloadEngine(
            modelPath = path,
            preferredAccel = prefs.acceleratorFlow(LocalRuntime.LiteRT).first(),
            forceCpu = prefs.forceCpu(LocalRuntime.LiteRT),
            maxNumTokens = tokens,
            supportImage = imageSupport,
            supportAudio = config.supportsAudio,
            visionAccelerator = config.visionAccelerator ?: "gpu",
        )
    }
}
