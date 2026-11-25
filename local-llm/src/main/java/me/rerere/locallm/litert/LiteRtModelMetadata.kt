package me.rerere.locallm.litert

import me.rerere.ai.provider.Modality
import me.rerere.ai.provider.ModelAbility
import me.rerere.ai.provider.Model
import com.google.ai.edge.litertlm.ModelInfo
import java.io.File

/**
 * Derives `Model.inputModalities` + `Model.abilities` for a LiteRT model file from the
 * curated [LiteRtModelDefaults] config. Mirrors how cloud models populate the same fields
 * via `ModelRegistry.MODEL_INPUT_MODALITIES` / `MODEL_ABILITIES` — but driven by the
 * LiteRT catalog instead of the cloud model definitions.
 *
 * For unknown files (HF-URL-pasted models that aren't in our catalog), falls back to the
 * safest assumption: TEXT-only input, TOOL ability (LiteRT-supported models are typically
 * tool-tuned; DeepSeek-R1-Distill was excluded for this reason).
 */
object LiteRtModelMetadata {

    // Native metadata may omit capabilities in legacy exports. Keep curated defaults
    // for those files rather than disabling features that already work.
    internal data class FileInfo(val ceiling: Int?, val vision: Boolean, val audio: Boolean, val thinking: Boolean, val tools: Boolean,
        val topK: Int, val topP: Double, val temperature: Double)
    private data class CachedInfo(val path: String, val size: Long, val modified: Long, val info: FileInfo?)
    @Volatile private var cachedInfo: CachedInfo? = null

    internal fun inspect(file: File): FileInfo? {
        val path = file.absolutePath
        val size = file.length()
        val modified = file.lastModified()
        cachedInfo?.takeIf { it.path == path && it.size == size && it.modified == modified }?.let { return it.info }
        val info = runCatching {
            ModelInfo.from(path).use { model ->
                val llm = model as? ModelInfo.Llm ?: return@use null
                val modalities = llm.inputModalities()
                val sampler = llm.defaultSamplerParams()
                FileInfo(llm.maxContextTokens().takeIf { it in 128..(1 shl 20) }, modalities.vision, modalities.audio,
                    llm.supportsThinking(), llm.supportsFunctionCalling(), sampler.topK, sampler.topP.toDouble(), sampler.temperature.toDouble())
            }
        }.getOrNull()
        cachedInfo = CachedInfo(path, size, modified, info)
        return info
    }

    internal fun runtimeConfig(file: File, modelId: String): LiteRtModelConfig {
        val base = LiteRtModelDefaults.forModelFile(modelId)
        val info = inspect(file) ?: return base
        val vision = base.supportsImage || info.vision
        val hasSampler = info.topK > 0 && info.topP in 0.0..1.0 && info.temperature.isFinite() && info.temperature >= 0.0
        return base.copy(
            supportsImage = vision,
            visionAccelerator = if (vision) base.visionAccelerator ?: "gpu" else null,
            supportsAudio = base.supportsAudio || info.audio,
            supportsThinking = base.supportsThinking || info.thinking,
            topK = if (hasSampler) info.topK else base.topK,
            topP = if (hasSampler) info.topP else base.topP,
            temperature = if (hasSampler) info.temperature else base.temperature,
        )
    }

    fun modelForFile(file: File, modelId: String = file.name): Model {
        val config = runtimeConfig(file, modelId)
        val capabilities = deriveCapabilities(modelId)
        val modalities = if (config.supportsImage) listOf(Modality.TEXT, Modality.IMAGE) else listOf(Modality.TEXT)
        val abilities = buildList {
            if (inspect(file)?.tools == true || ModelAbility.TOOL in capabilities.abilities) add(ModelAbility.TOOL)
            if (config.supportsThinking) add(ModelAbility.REASONING)
        }
        return Model(modelId = modelId, displayName = LiteRtCatalog.displayNameFor(modelId),
            inputModalities = modalities, abilities = abilities,
            contextLength = (LiteRtModelFile.resolveContextCeiling(file, modelId, config) ?: config.maxTokens).coerceAtMost(32768))
    }

    data class Capabilities(
        val inputModalities: List<Modality>,
        val abilities: List<ModelAbility>,
    )

    /**
     * Pure mapper: catalog file -> `(inputModalities, abilities)`. JVM-unit-testable.
     *
     * Mapping rules (matches [LiteRtModelConfig]):
     *   - `supportsImage = true`  -> add Modality.IMAGE.
     *   - `supportsThinking = true` -> add ModelAbility.REASONING.
     *   - ModelAbility.TOOL is always added (every model we ship through the LiteRT
     *     catalog is tool-tuned; the agent loop depends on it; DeepSeek-R1-Distill was
     *     deliberately removed from the catalog because it isn't tool-tuned).
     */
    fun deriveCapabilities(modelFile: String): Capabilities {
        val config = LiteRtModelDefaults.forModelFile(modelFile)
        val modalities = buildList {
            add(Modality.TEXT)
            if (config.supportsImage) add(Modality.IMAGE)
        }
        val abilities = buildList {
            add(ModelAbility.TOOL)
            if (config.supportsThinking) add(ModelAbility.REASONING)
        }
        return Capabilities(modalities, abilities)
    }

    /**
     * Additive merge for the startup migration: take the persisted model's current
     * capabilities and UNION them with the catalog's. Never removes anything the user
     * had — so a user who manually checked REASONING on a model the catalog says is
     * text-only keeps their override.
     *
     * Stable order: TEXT before IMAGE in modalities; TOOL before REASONING in abilities.
     */
    fun mergeAdditive(current: Capabilities, target: Capabilities): Capabilities {
        val modalitySet = current.inputModalities.toMutableSet().apply {
            addAll(target.inputModalities)
        }
        val abilitySet = current.abilities.toMutableSet().apply {
            addAll(target.abilities)
        }
        return Capabilities(
            inputModalities = listOf(Modality.TEXT, Modality.IMAGE).filter { it in modalitySet },
            abilities = listOf(ModelAbility.TOOL, ModelAbility.REASONING).filter { it in abilitySet },
        )
    }
}
