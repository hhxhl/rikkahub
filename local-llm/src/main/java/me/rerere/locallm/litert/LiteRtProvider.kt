package me.rerere.locallm.litert

import android.util.Log
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import me.rerere.ai.core.MessageRole
import me.rerere.ai.core.ReasoningLevel
import me.rerere.ai.provider.ImageGenerationParams
import me.rerere.ai.provider.Model
import me.rerere.ai.provider.Provider
import me.rerere.ai.provider.ProviderSetting
import me.rerere.ai.provider.TextGenerationParams
import me.rerere.ai.provider.TextGenerationResult
import me.rerere.ai.ui.ImageGenerationItem
import me.rerere.ai.ui.StreamChunk
import me.rerere.ai.ui.StreamChunkHandler
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import android.content.Context
import me.rerere.locallm.LocalRuntime
import me.rerere.locallm.LocalRuntimePreferences
import com.google.ai.edge.litertlm.tool as litertTool
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.ThinkingConfig

private const val TAG = "LiteRtProvider"

/** A local prompt did not fit; compression may retry with smaller complete batches. */
class LocalContextBudgetException(message: String, cause: Throwable? = null) : IllegalArgumentException(message, cause)

/**
 * Map RikkaHub's existing reasoning picker to the LiteRT-LM 0.19.1 native API.
 * AUTO inherits the model's default instead of forcing thinking on or off.
 * Positive budgets are the existing RikkaHub levels, not a new LiteRT setting.
 */
internal fun liteRtThinkingConfig(level: ReasoningLevel): ThinkingConfig? = when (level) {
    ReasoningLevel.AUTO -> null
    ReasoningLevel.OFF -> ThinkingConfig(enableThinking = false, thinkingTokenBudget = 0)
    else -> ThinkingConfig(enableThinking = true, thinkingTokenBudget = level.budgetTokens)
}

/** Apply the user-configured SYSTEM-message character cap before prompt budgeting. */
internal fun truncateLocalSystemText(text: String, charLimit: Int): String {
    require(charLimit > 0)
    return if (text.length > charLimit) {
        text.take(charLimit) + "\n…(truncated for local model context)"
    } else {
        text
    }
}

/**
 * Drop an entire oldest USER-led segment, up to (but not including) the next USER.
 * Returning the original list means the oldest segment cannot safely be trimmed.
 * In particular, never turn the latest assistant/tool result into an orphan input.
 */
internal fun <T> trimOldestConversationSegment(
    messages: List<T>,
    isUser: (T) -> Boolean,
): List<T> {
    if (messages.size <= 1) return messages
    val nextUser = (1 until messages.size).firstOrNull { isUser(messages[it]) }
        ?: return messages
    return messages.drop(nextUser)
}

/**
 * Outcome of [decideImageForwarding]: whether to hand image bytes to the runtime this turn,
 * and whether the user's attached images were dropped in a way that warrants a chat note.
 */
internal data class ImageForwardingDecision(
    val forwardImages: Boolean,
    val noteImagesDropped: Boolean,
)

/**
 * Decide whether the current turn's images may be forwarded to [LiteRtRuntime.streamTurns].
 *
 * The forward gate keys off [visionEnabledPostLoad] — the ACTUAL vision state of the loaded
 * engine ([LiteRtRuntime.LoadOutcome.visionEnabled]) — NOT the pre-load estimate. When the GPU
 * vision executor fails to initialise (Adreno 7xx + OEM ROMs; upstream LiteRT-LM #2292), the
 * runtime falls back to a text-only engine; forwarding image bytes to that engine null-derefs
 * the native runtime (SIGSEGV in `liblitertlm_jni.so`). Reading the pre-load estimate here was
 * the root cause of that crash.
 *
 * [ImageForwardingDecision.noteImagesDropped] is true only when the user actually attached
 * images to an image-capable model whose vision is not live — i.e. a vision model that could
 * not bring up its encoder on this device. A plain text-only model with a stray image
 * attachment keeps the prior silent drop (the device's vision capability was never the issue).
 */
internal fun decideImageForwarding(
    modelImageCapable: Boolean,
    visionEnabledPostLoad: Boolean,
    userSentImages: Boolean,
): ImageForwardingDecision = ImageForwardingDecision(
    forwardImages = visionEnabledPostLoad,
    noteImagesDropped = userSentImages && modelImageCapable && !visionEnabledPostLoad,
)

/**
 * Implements the existing Provider interface so any assistant can pick a LiteRT
 * model the same way it picks an OpenAI / Claude model. Routes inference through
 * [LiteRtRuntime] and tool calls through [LiteRtToolPrefix].
 *
 * When initialization reports possible model corruption or SDK incompatibility, retain
 * the original model file and configuration. Native errors alone are not proof that a
 * downloaded model is irrecoverably corrupt (a GPU/SDK mismatch can look identical).
 *
 * @param settingsUpdater Suspend lambda that lets the provider mutate the persisted
 *   [ProviderSetting] list without coupling the `local-llm` module to `SettingsStore`
 *   directly. Supply `{ fn -> settingsStore.update { old -> old.copy(providers = fn(old.providers)) } }`
 *   from the DI layer.
 */
class LiteRtProvider(
    private val context: Context,
    private val runtime: LiteRtRuntime,
    private val prefs: LocalRuntimePreferences,
    private val settingsUpdater: suspend (transform: (List<ProviderSetting>) -> List<ProviderSetting>) -> Unit,
) : Provider<ProviderSetting.LiteRtLocal> {

    /** Do not auto-delete user-owned files on an SDK/backend initialization error. */
    private fun handleCorruptModel(corrupt: LiteRtModelCorruptException): Nothing {
        val fileName = java.io.File(corrupt.modelPath).name
        Log.e(TAG, "Model load failed; original file and settings preserved: $fileName", corrupt)
        throw RuntimeException(
            "LiteRT could not load \"$fileName\". This may be a damaged file, " +
                "an incompatible export, or an SDK/backend issue. Your model file and " +
                "settings were NOT deleted. Check the model format and selected backend " +
                "before manually reinstalling it.",
            corrupt,
        )
    }

    override suspend fun listModels(providerSetting: ProviderSetting.LiteRtLocal): List<Model> {
        val installed = prefs.installedModels(LocalRuntime.LiteRT)
        return installed.mapNotNull { (fileName, path) ->
            java.io.File(path).takeIf { it.isFile }?.let { LiteRtModelMetadata.modelForFile(it, fileName) }
        }
    }

    override suspend fun generateText(
        providerSetting: ProviderSetting.LiteRtLocal,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): TextGenerationResult {
        var collected = listOf(UIMessage(role = MessageRole.ASSISTANT, parts = emptyList()))
        val handler = StreamChunkHandler(params.model)
        streamText(providerSetting, messages, params).collect { chunk ->
            collected = handler.handle(collected, chunk)
        }
        return TextGenerationResult(
            id = "litert-${System.currentTimeMillis()}",
            model = params.model.modelId,
            message = collected.last(),
        )
    }

    override suspend fun streamText(
        providerSetting: ProviderSetting.LiteRtLocal,
        messages: List<UIMessage>,
        params: TextGenerationParams,
    ): Flow<StreamChunk> = flow {
        runtime.withGenerationReservation {
        val installed = prefs.installedModels(LocalRuntime.LiteRT)
        val modelPath = installed[params.model.modelId]
            ?: throw IllegalStateException("Model ${params.model.modelId} not installed")

        // Guard against a stale DataStore entry (file registered but later deleted by the user,
        // ADB, or a failed partial download).  Surface a clean message rather than letting the
        // engine throw an opaque native error.
        if (!java.io.File(modelPath).exists()) {
            throw IllegalStateException(
                "Model file for \"${params.model.modelId}\" is no longer present on disk " +
                "($modelPath). Delete the model entry in Settings → Providers → Local · LiteRT " +
                "and re-download it."
            )
        }

        // Box-style model backend: a fixed explicit CPU/GPU/NPU choice, default GPU.
        // Never infer it from a previous Engine result or a global CPU fallback cache.
        val selectedBackend = prefs.modelAccelerator(LocalRuntime.LiteRT, params.model.modelId)

        // Per-model defaults curated to mirror Gallery's `model_allowlists/1_0_13.json`
        // (topK / topP / temperature / maxTokens / multimodal flags / speculative decoding).
        // For HF-pasted models that aren't in the table, this returns the FALLBACK config.
        val config = LiteRtModelMetadata.runtimeConfig(java.io.File(modelPath), params.model.modelId)
        // The model's KV cache ceiling, most specific source first:
        //   1. explicit exported `ekvNNNN` cache marker (if present),
        //   2. the metadata context length (conservative when export size is unknown),
        //   3. the curated table. A metadata default can be smaller than exported KV.
        // This is NOT advisory. `EngineConfig.maxNumTokens` is taken on trust by the
        // engine, so a value above the KV cache the file was built for makes prefill run
        // off the end of that cache and fault inside the native executor: SIGSEGV on the
        // execution thread, nothing catchable, process gone. The comment that used to sit
        // here claimed "the underlying KV cache size still caps it" — it does not, and
        // that assumption is what let a 32000-token setting reach a 2048-token model.
        val contextCeiling = LiteRtModelFile.resolveContextCeiling(
            java.io.File(modelPath), params.model.modelId, config,
        )

        // User-set max-context override: never beyond the identified file cache ceiling.
        val runtimeSettings = prefs.liteRtRuntimeSettings(LocalRuntime.LiteRT)
        val maxNumTokensOverride = runtimeSettings.maxNumTokensOverride
        val requestedMaxNumTokens = maxNumTokensOverride ?: contextCeiling?.coerceAtMost(32768) ?: config.maxTokens
        val effectiveMaxNumTokens = configuredContextTokens(config, contextCeiling, maxNumTokensOverride)
        require(promptByteBudget(effectiveMaxNumTokens, runtimeSettings.responseReservePercent) > 0) {
            "The selected LiteRT context leaves no input budget; increase context or reduce response reserve."
        }
        if (contextCeiling != null && requestedMaxNumTokens > contextCeiling) {
            Log.w(
                TAG,
                "clamped maxNumTokens $requestedMaxNumTokens -> $contextCeiling for " +
                    "${params.model.modelId}: the model cannot hold more than its declared " +
                    "ceiling, and asking for more faults the native executor",
            )
        }

        // ---- System instruction (RADICALLY TRIMMED for small-context local models) ----
        //
        // Cloud providers happily accept full system prompts: agent-core skill body
        // (~3-5k tokens of persona + tool docs) + every tool's JSON schema (~5-10k more)
        // fits comfortably inside a 32k–200k context window. LiteRT models max out at
        // 1-32k TOTAL, often 4k. A fresh "hi" against the default Qwen 4k model SDK-aborts
        // with "Input token ids are too long: 18031 >= 4096" because we forwarded the same
        // bulky system instruction the cloud path uses.
        //
        // For LiteRT we therefore:
        //  - Use [buildCompactPrefix] (one-line `- name: desc` per tool, NO schemas).
        //  - Drop bulky auto-loaded skill bodies — keep at most the user-configured
        //    system-message char limit (room for the user's
        //    custom system prompt + a short identity line, not for skill prose).
        //
        // A 1.5B model can't usefully consume agent-core's persona docs anyway — the
        // model lacks the capacity to follow that level of instruction.
        val systemTextsRaw = messages
            .filter { it.role == MessageRole.SYSTEM }
            .map { msg ->
                msg.parts.filterIsInstance<UIMessagePart.Text>().joinToString("") { it.text }
            }
            .filter { it.isNotBlank() }
            .joinToString("\n\n")
        val systemMessageCharLimit = runtimeSettings.systemMessageCharLimit
        val trimmedSystemTexts = truncateLocalSystemText(systemTextsRaw, systemMessageCharLimit)

        // Native tool calling. Every declared tool is handed to the runtime under its real
        // name with its real JSON schema (see [LiteRtToolDeclaration]), so the model calls
        // `web_fetch(url=...)` rather than a generic dispatcher: the model's own chat
        // template renders the declarations, which is what these models were tuned on.
        //
        // The conversation sets `automaticToolCalling = false`, so the runtime hands the
        // calls back and we republish them as ordinary tool parts. That keeps local models
        // on the same execution path as every cloud provider: HARDLINE floor, approval
        // prompt, auto-approve allowlist, and the per-turn wall-clock budget all apply.
        // No system-prompt tool catalogue is needed, and none is added: a duplicate
        // listing would only compete with the template's own rendering for context.
        //
        // Declarations are budgeted in UTF-8 bytes against the actual Engine allocation,
        // leaving room for dialogue, a response, and the template. A tool that cannot fit
        // is skipped intact rather than truncating its JSON schema.
        val contextTokens = effectiveMaxNumTokens
        val combinedSystem = trimmedSystemTexts.trim()
        val systemBytes = combinedSystem.toByteArray(Charsets.UTF_8).size
        val reservePercent = runtimeSettings.responseReservePercent
        val toolThreshold = runtimeSettings.toolEnableThresholdTokens
        val toolByteBudget = toolDeclarationByteBudget(
            contextTokens, systemBytes, reservePercent, toolThreshold,
        )
        var toolBytesUsed = 0
        val droppedTools = mutableListOf<String>()
        val declarations = params.tools.mapNotNull { tool ->
            val declaration = LiteRtToolDeclaration(tool)
            val cost = declaration.declarationJson.toByteArray(Charsets.UTF_8).size + TOOL_OVERHEAD_BYTES
            if (toolBytesUsed + cost > toolByteBudget) {
                droppedTools += tool.name
                null
            } else {
                toolBytesUsed += cost
                declaration
            }
        }
        val nativeTools: List<ToolProvider> = declarations.map { litertTool(it) }
        if (params.tools.isNotEmpty()) {
            Log.i(
                TAG,
                "declared ${nativeTools.size}/${params.tools.size} tool(s) natively, " +
                    "$toolBytesUsed/$toolByteBudget bytes " +
                    "(context ${contextTokens}t, threshold ${toolThreshold}t)",
            )
        }
        // Never drop tools silently: an assistant with a large enabled set on a small model
        // loses real capability here, and that has to be visible when debugging why the
        // model "ignored" a tool.
        if (droppedTools.isNotEmpty()) {
            Log.w(
                TAG,
                "dropped ${droppedTools.size} tool(s) over the declaration budget: " +
                    droppedTools.joinToString(",").take(300),
            )
        }

        // ---- Conversation history → turn list + cold-path blob ----
        //
        // We hand the runtime BOTH:
        //   - `turns`: the full (trimmed) turn list, each carrying a content signature. The
        //     runtime uses these to decide whether its warm Conversation's KV cache already
        //     holds a prefix of this history — if so it sends ONLY the new turn and reuses
        //     the cache (Gallery's behaviour), instead of re-prefilling the whole history.
        //   - `coldBlob`: a compatibility fallback for non-user-ending agent/tool turns.
        //     Ordinary cold USER turns restore prior roles via SDK `initialMessages`.
        //
        // TOOL-role messages are dropped here: their content is already represented inline
        // in the immediately-preceding ASSISTANT message's Tool part output.
        //
        // Trimming: adapt to the configured Engine context, actual tool declarations,
        // system bytes and reserved response capacity. Drop whole older USER-led segments;
        // never leave an orphan ASSISTANT reply at the start or truncate the latest input.
        var trimmed = messages.filter {
            it.role != MessageRole.SYSTEM && it.role != MessageRole.TOOL
        }
        val untrimmedTurnCount = trimmed.size
        val historyBudget = historyByteBudget(contextTokens, systemBytes, toolBytesUsed, reservePercent)
        var coldBlob = renderColdBlob(trimmed, historyBudget)
        while (coldBlob.toByteArray(Charsets.UTF_8).size > historyBudget && trimmed.size > 1) {
            // Keep the existing summary; dropping it would forget all earlier turns.
            val checkpoint = trimmed.firstOrNull()?.takeIf { it.isContextCheckpoint }
            val tail = if (checkpoint == null) trimmed else trimmed.drop(1)
            val next = trimOldestConversationSegment(tail) { it.role == MessageRole.USER }
            if (next.size == tail.size) break
            trimmed = if (checkpoint == null) next else listOf(checkpoint) + next
            coldBlob = renderColdBlob(trimmed, historyBudget)
        }
        if (trimmed.size != untrimmedTurnCount) {
            Log.i(TAG, "history trimmed: $untrimmedTurnCount -> ${trimmed.size} turns; " +
                "dropped=${untrimmedTurnCount - trimmed.size}")
        }
        val historyBytes = coldBlob.toByteArray(Charsets.UTF_8).size
        if (historyBytes > historyBudget) {
            throw LocalContextBudgetException(
                "Latest message or retained summary exceeds the safe local context budget ($historyBytes > " +
                    "$historyBudget UTF-8 bytes for ${contextTokens} tokens). " +
                    "Shorten the message, reduce enabled tools, or use a larger context model."
            )
        }
        val turns = trimmed.map { it.toTurn(historyBudget) }
        // The final ASSISTANT message contains the host's executed tool parts.
        // Preserve call ids and arguments for exact matching against SDK tool calls.
        // Earlier tool parts in the same assistant turn are harmless: the runtime only
        // selects ids from the currently pending batch.
        val toolResults = trimmed.lastOrNull()
            ?.takeIf { it.role == MessageRole.ASSISTANT }
            ?.parts?.filterIsInstance<UIMessagePart.Tool>()
            ?.filter { it.isExecuted }
            ?.mapNotNull { part ->
                val rawText = part.output.filterIsInstance<UIMessagePart.Text>()
                    .joinToString("") { it.text }
                if (rawText.isBlank()) null else RuntimeToolResult(
                    id = part.toolCallId,
                    name = part.toolName,
                    argumentsJson = part.input.ifBlank { "{}" },
                    output = compactLocalToolOutput(part.toolName, rawText, toolOutputByteLimit(trimmed.last(), historyBudget)),
                )
            }.orEmpty()

        // Log actual prompt byte usage. The heuristic budget is intentionally conservative:
        // LiteRT-LM does not expose an exact prompt-token counter through this API.
        val prefillBytes = systemBytes + toolBytesUsed + historyBytes
        Log.i(
            TAG,
            "prefill budget: system=${systemBytes}B tools=${toolBytesUsed}B " +
                "history=${historyBytes}B total=${prefillBytes}B " +
                "(input byte cap=${promptByteBudget(contextTokens, reservePercent)}B, " +
                    "reserve=${reservePercent}%, engine=${contextTokens}t)",
        )

        // Match Box: use the model's explicit vision configuration. A failed init is
        // reported instead of silently changing the model's capabilities.
        val effectiveSupportImage = config.supportsImage && config.visionAccelerator != null

        // ---- Engine load with full per-model + per-call config ----
        val requestId = java.util.UUID.randomUUID().toString()
        val nativeThinking = liteRtThinkingConfig(params.reasoningLevel)
        Log.i(TAG, "reasoning level=${params.reasoningLevel}, " +
            "thinking=${nativeThinking?.enableThinking ?: "model-default"}, " +
            "budget=${nativeThinking?.thinkingTokenBudget ?: "model-default"}")
        val outcome = try {
            runtime.ensureLoaded(
                modelPath = modelPath,
                requestId = requestId,
                preferredAccel = selectedBackend,
                maxNumTokens = effectiveMaxNumTokens,
                supportImage = effectiveSupportImage,
                supportAudio = config.supportsAudio,
                // Match Google AI Edge Gallery: speculative decoding is an explicit user
                // opt-in (their `ConfigKeys.ENABLE_SPECULATIVE_DECODING` defaults to false).
                // Forcing it on whenever the model file supports it has been a candidate
                // contributor to GPU init regressions on Adreno-class devices; matching
                // Gallery removes that variable. Re-enable if/when we expose a settings
                // toggle and the user explicitly opts in.
                speculativeDecoding = false,
                visionAccelerator = config.visionAccelerator ?: "gpu",
                systemInstructionText = combinedSystem.ifBlank { null },
                tools = nativeTools,
                toolDeclarations = declarations.map { it.declarationJson },
                thinkingConfig = nativeThinking,
                // Constrained decoding ON when this turn carries tools — matches Gallery's
                // tool-task behaviour. The SDK projects the model's output into the @Tool
                // call grammar so Gemma actually invokes runTool(...) via the bridge
                // instead of producing free-form "I can't do that" text. OFF for plain
                // chat turns to avoid biasing normal language.
                constrainedDecoding = nativeTools.isNotEmpty(),
                topK = config.topK,
                topP = params.topP?.toDouble() ?: config.topP,
                temperature = params.temperature?.toDouble() ?: config.temperature,
                initialTurnHistory = turns,
            )
        } catch (corrupt: LiteRtModelCorruptException) {
            handleCorruptModel(corrupt)
        } catch (t: Throwable) {
            throw translateSdkError(t, effectiveMaxNumTokens)
        }
        try {
        // Box-style: do not persist the resolved backend or an implicit fallback.
        // The user's per-model backend preference is the only source of truth.

        // ---- Stream + cumulative→delta conversion ----
        //
        // The runtime accumulates the SDK's onMessage chunks and emits the
        // CUMULATIVE response so far. But
        // RikkaHub's downstream chunk-handling APPENDS each Text part (see
        // UIMessage.appendChunk in ai/ui/Message.kt: incoming Text parts are concatenated
        // onto the previous Text part). Emitting cumulative chunks straight through would
        // duplicate the response geometrically. So we hold the previous cumulative and
        // emit only the new suffix as the delta.
        //
        // We also collect the full cumulative response so we can extract tool-call blocks
        // at end-of-stream (see the comment block right after the collect{} below for why
        // we do this once at the end rather than incrementally like AICoreProvider does).
        val streamId = "litert-${System.currentTimeMillis()}"
        // One stable id for the whole turn's prose (the vision-dropped note, if any, plus
        // every generation delta) so StreamChunkHandler appends them into a single Text part,
        // matching the old positional-append merge behaviour.
        val textId = "$streamId-text"
        var previousCumulative = ""
        var reasoningSequence = 0
        var activeReasoningId: String? = null
        suspend fun endReasoning() {
            activeReasoningId?.let { emit(StreamChunk.ReasoningEnd(it)) }
            activeReasoningId = null
        }
        val fullResponseBuilder = StringBuilder()

        // Extract images + audio from the LAST USER message only (the new turn). Earlier
        // turns' media is already represented in the conversation prefix via the cold-blob
        // text render; the SDK's vision/audio executors only consume current-turn input.
        // Decode here so the runtime gets ready-to-use Bitmaps / encoded audio bytes and
        // doesn't have to know about content URIs.
        val lastUser = trimmed.lastOrNull()?.takeIf { it.role == MessageRole.USER }
        val userImageParts = lastUser?.parts
            ?.filterIsInstance<UIMessagePart.Image>()
            .orEmpty()
        // Gate image forwarding on the ACTUAL post-load vision state (outcome.visionEnabled),
        // NOT the pre-load effectiveSupportImage. ensureLoaded may have fallen back to a
        // text-only engine when the GPU vision executor failed to init; forwarding image bytes
        // to an engine with no vision executor crashes the native runtime (SIGSEGV in
        // liblitertlm_jni.so). See [decideImageForwarding].
        val imageDecision = decideImageForwarding(
            modelImageCapable = config.supportsImage && config.visionAccelerator != null,
            visionEnabledPostLoad = outcome.visionEnabled,
            userSentImages = userImageParts.isNotEmpty(),
        )
        val turnImages: List<android.graphics.Bitmap> = if (imageDecision.forwardImages) {
            userImageParts.mapNotNull { it.toLiteRtBitmap(context) }
        } else {
            // Vision unavailable / not supported by this model — drop image parts. When the
            // user actually attached images to a vision-capable model, the note below tells
            // them why; otherwise it's a silent no-op (text-only model, no attachment).
            emptyList()
        }
        val turnAudio: List<ByteArray> = if (config.supportsAudio) {
            lastUser?.parts
                ?.filterIsInstance<UIMessagePart.Audio>()
                ?.mapNotNull { it.toLiteRtAudioBytes(context) }
                .orEmpty()
        } else {
            emptyList()
        }
        if (turnImages.isNotEmpty() || turnAudio.isNotEmpty()) {
            Log.i(
                TAG,
                "streamText: forwarding ${turnImages.size} image(s) + ${turnAudio.size} " +
                    "audio clip(s) to runtime for modelId=${params.model.modelId}",
            )
        }

        // The user attached image(s) to a vision-capable model whose encoder couldn't
        // initialise on this device, so they were dropped above. Tell the user once, in-line,
        // so a text-only reply to an image prompt isn't silently confusing.
        if (imageDecision.noteImagesDropped) {
            Log.w(
                TAG,
                "vision not live post-load for ${params.model.modelId}; dropped " +
                    "${userImageParts.size} image(s), noting it to the user",
            )
            emit(StreamChunk.TextDelta(textId, VISION_DROPPED_NOTE))
        }

        var toolCallCount = 0
        var contextTokenCount: Int? = null
        run {
            try {
                runtime.streamTurns(
                    requestId = requestId,
                    history = turns,
                    coldBlob = coldBlob,
                    toolResults = toolResults,
                    images = turnImages,
                    audioClips = turnAudio,
                    maxOutputToken = params.maxTokens?.takeIf { it > 0 }?.coerceAtMost(effectiveMaxNumTokens),
                ).collect { event ->
                    when (event) {
                        is StreamEvent.ThinkingDelta -> {
                            val id = activeReasoningId ?: "$streamId-thought-${reasoningSequence++}".also {
                                activeReasoningId = it
                                emit(StreamChunk.ReasoningStart(it))
                            }
                            emit(StreamChunk.ReasoningDelta(id, event.text))
                        }
                        is StreamEvent.Delta -> {
                            val cumulative = event.cumulative
                            // Defensive: if the SDK ever emits a non-monotonic cumulative (e.g.
                            // after an internal retry or template re-tokenisation), treat the new
                            // payload as a fresh start, emitting it whole as the delta rather than
                            // computing a negative-length suffix that would silently drop
                            // characters.
                            val delta = if (cumulative.startsWith(previousCumulative)) {
                                cumulative.substring(previousCumulative.length)
                            } else {
                                cumulative
                            }
                            previousCumulative = cumulative
                            fullResponseBuilder.setLength(0)
                            fullResponseBuilder.append(cumulative)

                            if (delta.isNotEmpty()) {
                                endReasoning()
                                emit(StreamChunk.TextDelta(textId, delta))
                            }
                        }

                        is StreamEvent.ContextTokens -> {
                            contextTokenCount = event.count
                        }

                        is StreamEvent.ToolCalls -> {
                            endReasoning()
                            // IDs are minted by the runtime and retained until RikkaHub
                            // finishes approval/execution. This allows a later native
                            // Message.tool response to match exactly the original call.
                            event.calls.forEach { call ->
                                toolCallCount++
                                Log.i(TAG, "model requested tool '${call.name}' (id=${call.id})")
                                emit(StreamChunk.ToolCallStart(id = call.id, toolName = call.name))
                                emit(StreamChunk.ToolCallDelta(id = call.id, inputDelta = call.argumentsJson))
                            }
                        }
                    }
                }
            } catch (t: Throwable) {
                // Translate the LiteRT-LM SDK's raw native-error text into something the user
                // and the chat error surface can act on. The most common one — "Input token
                // ids are too long. Exceeding the maximum number of tokens allowed: N >= M"
                // — looks scary but means "your conversation is bigger than this model can
                // hold". Surface that with a recovery hint.
                throw translateSdkError(t, effectiveMaxNumTokens)
            }
        }

        // Close the stream. Tool calls were republished as tool parts above and have NOT
        // run yet: the host executes them after applying approval and the HARDLINE floor,
        // then calls back in with the results. Reporting "tool_calls" is what tells the
        // agent loop to do that; reporting "stop" would end the turn with the model's
        // request silently dropped.
        endReasoning()
        emit(StreamChunk.Finish(
            finishReason = if (toolCallCount > 0) "tool_calls" else "stop",
            contextTokenCount = contextTokenCount,
            contextHistoryTrimmed = trimmed.size != untrimmedTurnCount,
        ))
        } finally {
            // Cancellation must also release the ensureLoaded→streamTurns reservation.
            kotlinx.coroutines.withContext(kotlinx.coroutines.NonCancellable) {
                runtime.releasePendingStream(modelPath, requestId)
            }
        }
        }
    }

    /**
     * Render ONE message's content WITHOUT a role marker — the raw turn text.
     *
     * Used for the warm path and native initialMessages, with the SDK's chat template
     * applying actual USER/MODEL roles (as in Gallery). Also retained for the tool
     * continuation compatibility fallback [renderHistoryAsChatML]. Tool calls/outputs are kept in the
     * `<tool_call>…</tool_call>` / `<tool_result>…</tool_result>` shape the prompt prefix
     * teaches the model, preserving prompt-engineered tool-calling continuity across turns.
     */
    private fun toolOutputByteLimit(message: UIMessage, historyBudget: Int): Int =
        (historyBudget / 2 / maxOf(1, message.getTools().count { it.isExecuted })).coerceAtLeast(1)

    private fun renderTurnRawText(msg: UIMessage, historyBudget: Int): String = buildString {
        when (msg.role) {
            MessageRole.USER -> {
                append(
                    msg.parts.filterIsInstance<UIMessagePart.Text>().joinToString("") { it.text }
                )
            }
            MessageRole.ASSISTANT -> {
                // Walk the parts in order so an assistant turn that interleaves text +
                // tool calls is reconstructed faithfully.
                for (part in msg.parts) {
                    when (part) {
                        is UIMessagePart.Text -> append(part.text)
                        is UIMessagePart.Notice -> append(part.text)
                        is UIMessagePart.Tool -> {
                            val callJson = """{"name":"${part.toolName}","arguments":${part.input.ifBlank { "{}" }}}"""
                            append("<tool_call>").append(callJson).append("</tool_call>")
                            if (part.isExecuted) {
                                val outText = part.output
                                    .filterIsInstance<UIMessagePart.Text>()
                                    .joinToString("") { it.text }
                                if (outText.isNotEmpty()) {
                                    append("\n<tool_result>")
                                        .append(compactLocalToolOutput(part.toolName, outText, toolOutputByteLimit(msg, historyBudget)))
                                        .append("</tool_result>")
                                }
                            }
                        }
                        else -> { /* image/audio/document parts not yet wired into LiteRT input */ }
                    }
                }
            }
            else -> { /* SYSTEM / TOOL are filtered out by the caller */ }
        }
    }

    /** Convert a message into a [Turn] for the runtime's warm/cold KV-cache decision. */
    private fun UIMessage.toTurn(historyBudget: Int): Turn {
        val turnRole = when (this.role) {
            MessageRole.USER -> ROLE_USER
            MessageRole.ASSISTANT -> ROLE_ASSISTANT
            else -> "other"
        }
        val replayParts = if (this.role == MessageRole.ASSISTANT) {
            parts.mapNotNull { part ->
                when (part) {
                    is UIMessagePart.Text -> NativeReplayPart.Text(part.text)
                    is UIMessagePart.Notice -> NativeReplayPart.Text(part.text)
                    is UIMessagePart.Tool -> {
                        val outputText = part.output.filterIsInstance<UIMessagePart.Text>()
                            .joinToString("") { it.text }
                        NativeReplayPart.Tool(
                            name = part.toolName,
                            argumentsJson = part.input.ifBlank { "{}" },
                            output = if (part.isExecuted && outputText.isNotBlank()) {
                                compactLocalToolOutput(part.toolName, outputText, toolOutputByteLimit(this, historyBudget))
                            } else null,
                        )
                    }
                    else -> null
                }
            }
        } else emptyList()
        return Turn(turnRole, renderTurnRawText(this, historyBudget), replayParts)
    }

    /**
     * The cold-path render of the (already SYSTEM/TOOL-filtered) history — what the runtime
     * sends when it cannot reuse a warm KV cache.
     *
     * For ordinary conversations, this is used for budgeting only: the runtime now sends
     * native initialMessages + the latest user message. For non-user-ending tool continuations,
     * keep the previous text fallback until native tool-result replay is implemented.
     */
    private fun renderColdBlob(messages: List<UIMessage>, historyBudget: Int): String =
        if (messages.size == 1) renderTurnRawText(messages[0], historyBudget)
        else renderHistoryAsChatML(messages, historyBudget)

    private fun renderHistoryAsChatML(messages: List<UIMessage>, historyBudget: Int): String = buildString {
        var first = true
        for (msg in messages) {
            if (msg.role == MessageRole.SYSTEM || msg.role == MessageRole.TOOL) continue
            if (!first) append("\n")
            first = false
            when (msg.role) {
                MessageRole.USER -> append("User: ").append(renderTurnRawText(msg, historyBudget))
                MessageRole.ASSISTANT -> append("Assistant: ").append(renderTurnRawText(msg, historyBudget))
                else -> { /* unreachable — filtered above */ }
            }
        }
        // Trailing assistant cue gives the model a clear "your turn" signal even after the
        // chat template wraps the input. Cheap belt-and-braces vs. the model going silent.
        if (isNotEmpty()) append("\nAssistant: ")
    }

    override suspend fun generateImage(
        providerSetting: ProviderSetting,
        params: ImageGenerationParams,
    ): Flow<ImageGenerationItem> = error("LiteRT does not support image generation in 22A")

    /**
     * Translate the LiteRT-LM SDK's raw native error message into a user-actionable
     * exception. The SDK throws bare `RuntimeException` with messages like
     * "Status Code: 3. Message: Input token ids are too long. Exceeding the maximum
     * number of tokens allowed: 4610 >= 4096" — technically correct but unhelpful for
     * users on a chat surface.
     */
    private fun translateSdkError(t: Throwable, effectiveMaxNumTokens: Int): Throwable {
        // An unavailable vision executor now surfaces as a load error. Do not imply
        // a text-only or CPU retry was attempted: Box-style mode does not perform one.
        if (t is LiteRtVisionUnavailableException) {
            return RuntimeException(
                "LiteRT 视觉后端初始化失败。所选后端不会自动切换；可尝试兼容的纯文本模型。" +
                    "原因：${t.cause?.message ?: t.message}",
                t,
            )
        }
        // Walk the SDK's exception cause chain to expose context-length errors.
        val joined = generateSequence<Throwable>(t) { it.cause }
            .map { it.message.orEmpty() }
            .joinToString("\n")

        // Context-length overflow. The numbers in the message tell us the actual
        // input vs. the limit; surface both plus what to do next.
        val overflow = Regex("""(\d+)\s*>=\s*(\d+)""").find(joined)
        if (joined.contains("Input token ids are too long", ignoreCase = true) ||
            (joined.contains("Status Code: 3", ignoreCase = true) && overflow != null)) {
            val (have, limit) = overflow?.let { it.groupValues[1] to it.groupValues[2] }
                ?: ("?" to effectiveMaxNumTokens.toString())
            return LocalContextBudgetException(
                "This model can only hold $limit tokens of context, but the conversation " +
                    "needs $have. Tap /new to start a fresh chat, switch to a model with a " +
                    "larger context (e.g. Gemma 4 E2B = 32k), OR raise \"Max context\" in " +
                    "Settings → Local · LiteRT if your model file supports it.",
                t,
            )
        }

        // Vision-executor failure. The SDK's stack trace points at
        // `vision_litert_compiled_model_executor.cc` when the model has vision
        // tensors but the runtime couldn't allocate the GPU vision encoder. The
        // pre-flight check covers force-CPU, but this catches the case where the
        // user has GPU enabled and it STILL fails (older device, GPU compute path
        // not supported for the model's vision config). Recovery: pick text-only.
        if (joined.contains("vision_litert_compiled_model_executor", ignoreCase = true) ||
            joined.contains("vision_litert", ignoreCase = true)) {
            return RuntimeException(
                "This model is multimodal (vision) and the GPU vision executor failed " +
                    "to initialise on this device. The most reliable workaround on devices " +
                    "without compatible GPU compute is to pick a text-only model: " +
                    "Gemma3-1B-IT or Qwen2.5-1.5B-Instruct " +
                    "from the catalog in Settings → Local · LiteRT.",
                t,
            )
        }

        // Anything else — keep the original; GenerationHandler's wrapper already
        // strips the stack trace from the LLM-facing envelope.
        return t
    }

    companion object {
        /** In-chat note shown once when the user attached image(s) to a vision-capable model
         *  whose vision encoder could not initialise on this device (text-only fallback). */
        private const val VISION_DROPPED_NOTE =
            "[Note: this model's vision encoder is unavailable on this device, so attached " +
                "images were not analysed. Replying from text only.]\n\n"

        /** Per-tool allowance for chat-template framing that is not present in JSON. */
        private const val TOOL_OVERHEAD_BYTES = 48

        /** Conservative byte budget: reserve the configured share (at least 256t) for
         *  output and another 96t for chat-template/native framing. UTF-8 bytes / 2 approximates an upper
         *  token count for multilingual text more safely than 4 characters per token.
         *  This is still a heuristic, not a replacement for SDK tokenization. */
        internal fun promptByteBudget(contextTokens: Int, responseReservePercent: Int = 25): Int {
            require(responseReservePercent in 10..75)
            val response = maxOf(
                256,
                (contextTokens.toLong() * responseReservePercent / 100).toInt(),
            )
            val available = (contextTokens - response - 96).coerceAtLeast(0)
            return (available.toLong() * 2).coerceAtMost(Int.MAX_VALUE.toLong()).toInt()
        }

        /** Keep at least half of the remaining input bytes for actual dialogue.
         *  The user-configurable threshold defaults to 1280 to preserve the original
         *  protection for tiny models; 0 disables only that cutoff, not byte budgeting. */
        internal fun toolDeclarationByteBudget(
            contextTokens: Int,
            systemBytes: Int,
            responseReservePercent: Int = 25,
            toolEnableThresholdTokens: Int = 1280,
        ): Int {
            require(toolEnableThresholdTokens in 0..32768)
            if (toolEnableThresholdTokens > 0 && contextTokens <= toolEnableThresholdTokens) return 0
            val remaining = (promptByteBudget(contextTokens, responseReservePercent) - systemBytes).coerceAtLeast(0)
            return remaining / 2
        }

        internal fun historyByteBudget(
            contextTokens: Int,
            systemBytes: Int,
            toolBytes: Int,
            responseReservePercent: Int = 25,
        ): Int = (promptByteBudget(contextTokens, responseReservePercent) - systemBytes - toolBytes).coerceAtLeast(0)

        /**
         * The context the engine is actually configured with: what was asked for, clamped to
         * the ceiling the model file can hold. Every part of the prefill is budgeted against
         * this one number, so the prompt can never be built for a context larger than the
         * engine that receives it.
         */
        internal fun configuredContextTokens(config: LiteRtModelConfig, ceiling: Int?, override: Int?): Int {
            val safeCeiling = (ceiling ?: config.maxTokens).coerceAtMost(32768)
            return minOf(override ?: safeCeiling, safeCeiling)
        }

        internal fun engineContextTokens(requestedMaxTokens: Int, contextCeiling: Int?): Int =
            contextCeiling?.let { minOf(requestedMaxTokens, it) } ?: requestedMaxTokens
    }
}
