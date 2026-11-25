package me.rerere.locallm.litert

import android.content.Context
import android.graphics.Bitmap
import android.os.Build
import android.os.PerformanceHintManager
import android.os.Process
import com.google.ai.edge.litertlm.Backend
import com.google.ai.edge.litertlm.Content
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Conversation
import com.google.ai.edge.litertlm.ConversationConfig
import com.google.ai.edge.litertlm.Engine
import com.google.ai.edge.litertlm.EngineConfig
import com.google.ai.edge.litertlm.ExperimentalApi
import com.google.ai.edge.litertlm.ExperimentalFlags
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.MessageCallback
import com.google.ai.edge.litertlm.ModelInfo
import com.google.ai.edge.litertlm.SamplerConfig
import com.google.ai.edge.litertlm.ToolProvider
import com.google.ai.edge.litertlm.ToolCall
import com.google.ai.edge.litertlm.Role
import com.google.ai.edge.litertlm.ThinkingConfig
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.channels.trySendBlocking
import me.rerere.ai.ui.UIMessagePart
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.withTimeoutOrNull
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.channelFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.CancellationException
import java.util.concurrent.TimeUnit
import me.rerere.locallm.LocalRuntime
import me.rerere.locallm.LocalRuntimePreferences

/** Role labels used in [Turn] signatures. Kept as plain strings so [turnSignature] is a
 *  pure function the provider and the runtime can both call without sharing an enum. */
const val ROLE_USER = "user"
const val ROLE_ASSISTANT = "assistant"

/**
 * Stable, content-derived signature for one conversation turn. Used to decide whether the
 * warm [Conversation]'s KV cache can be reused (see [planTurns]). Length + hashCode keeps
 * the collision rate negligible; a collision would only cause a (correct) cold reload, never
 * wrong output.
 */
fun turnSignature(role: String, text: String): String = "$role|${text.length}|${text.hashCode()}"

/**
 * One rendered conversation turn handed from [LiteRtProvider] to [LiteRtRuntime].
 *
 * [rawText] is the turn's content WITHOUT any "User:/Assistant:" markers — when this turn is
 * the single new turn appended to a warm Conversation, [rawText] is sent as-is and the SDK's
 * chat template applies role wrapping. Cold USER turns restore earlier turns as native
 * SDK Message.user/model items in ConversationConfig.initialMessages.
 */
/** Preserve the host's structured assistant content for native SDK Cold replay.
 * Do not infer tool calls from arbitrary assistant text containing <tool_call> tags.
 */
sealed interface NativeReplayPart {
    data class Text(val text: String) : NativeReplayPart
    data class Tool(val name: String, val argumentsJson: String, val output: String?) : NativeReplayPart
}

data class Turn(
    val role: String,
    val rawText: String,
    val nativeParts: List<NativeReplayPart> = emptyList(),
    val images: List<UIMessagePart.Image> = emptyList(),
    val audio: List<UIMessagePart.Audio> = emptyList(),
) {
    val signature: String get() = buildString {
        append(turnSignature(role, rawText))
        images.forEach { append("|image:${it.url.length}:${it.url.hashCode()}") }
        audio.forEach { append("|audio:${it.url.length}:${it.url.hashCode()}") }
    }
}

/**
 * One tool call the model asked for, lifted out of the SDK's `ToolCall` so callers do not
 * have to depend on litertlm types. [argumentsJson] is the arguments object serialised back
 * to a JSON string, matching the `input` field of `UIMessagePart.Tool`.
 */
data class RuntimeToolCall(val id: String, val name: String, val argumentsJson: String)

/** Executed by RikkaHub (including approval/denial), not by LiteRT-LM. */
data class RuntimeToolResult(
    val id: String,
    val name: String,
    val argumentsJson: String,
    val output: String,
)

/** SDK tool calls awaiting their corresponding host-approved results. */
internal data class PendingToolBatch(
    val anchorSignatures: List<String>,
    val calls: List<RuntimeToolCall>,
)

/**
 * The SDK has finished a native TOOL reply, but RikkaHub has not yet supplied the
 * final assistant message back to this Provider. RikkaHub stores tool calls, results
 * and subsequent model text inside ONE assistant UIMessage; the SDK stores them as
 * MODEL -> TOOL -> MODEL messages. Never guess a processed signature from the SDK
 * output alone: wait for the next request and check RikkaHub's *actual* history.
 */
internal data class CompletedNativeToolTurn(
    val precedingTurns: List<Turn>,
    val assistantBeforeFinalReply: String,
    val finalReply: String,
)

/**
 * Reconcile the completed SDK tool turn with RikkaHub's returned conversation.
 * A valid follow-up must be exactly [unchanged prefix, completed ASSISTANT, new USER].
 * Compare full raw text (not only hashes) to fail closed on edits, transformed output,
 * inserted notices, removed tool results or context trimming. Cold recovery remains
 * available when the model's visible reply differs from its raw SDK callback.
 */
internal fun reconciledToolTurnSignatures(
    completed: CompletedNativeToolTurn?,
    history: List<Turn>,
): List<String>? {
    if (completed == null || completed.finalReply.isBlank()) return null
    if (history.size != completed.precedingTurns.size + 2) return null
    if (history.last().role != ROLE_USER) return null
    if (history.dropLast(2) != completed.precedingTurns) return null
    val assistant = history[history.lastIndex - 1]
    if (assistant.role != ROLE_ASSISTANT) return null
    if (assistant.rawText != completed.assistantBeforeFinalReply + completed.finalReply) return null
    return history.dropLast(1).map { it.signature }
}

/**
 * Match executed host results to the *exact* SDK calls from the preceding generation.
 * Call ids matter: two invocations of the same function must not be confused. Fail
 * closed on edits, partial approvals, duplicate ids or argument changes.
 */
internal fun matchNativeToolResults(
    pending: PendingToolBatch?,
    history: List<Turn>,
    supplied: List<RuntimeToolResult>,
): List<RuntimeToolResult>? {
    if (pending == null || pending.calls.isEmpty() || history.lastOrNull()?.role != ROLE_ASSISTANT) return null
    if (history.dropLast(1).map { it.signature } != pending.anchorSignatures) return null
    if (supplied.map { it.id }.distinct().size != supplied.size) return null
    val byId = supplied.associateBy { it.id }
    return pending.calls.map { call ->
        val result = byId[call.id] ?: return null
        if (call.name != result.name || !sameToolArguments(call.argumentsJson, result.argumentsJson)) return null
        if (result.output.isBlank()) return null
        result
    }
}

private fun sameToolArguments(a: String, b: String): Boolean = runCatching {
    Json.parseToJsonElement(a) == Json.parseToJsonElement(b)
}.getOrDefault(false)

/** Diagnostic only: never loosen [matchNativeToolResults] to force a tool execution. */
internal fun nativeToolMatchFailureReason(
    pending: PendingToolBatch?,
    history: List<Turn>,
    supplied: List<RuntimeToolResult>,
): String {
    if (pending == null) return "pending-batch-missing"
    if (pending.calls.isEmpty()) return "pending-batch-empty"
    if (history.lastOrNull()?.role != ROLE_ASSISTANT) return "history-not-assistant-ending"
    if (history.dropLast(1).map { it.signature } != pending.anchorSignatures) {
        return "history-prefix-changed-or-trimmed"
    }
    if (supplied.map { it.id }.distinct().size != supplied.size) return "duplicate-result-id"
    val byId = supplied.associateBy { it.id }
    for (call in pending.calls) {
        val result = byId[call.id] ?: return "result-id-missing"
        if (call.name != result.name) return "tool-name-changed"
        if (!sameToolArguments(call.argumentsJson, result.argumentsJson)) return "tool-arguments-changed"
        if (result.output.isBlank()) return "empty-result"
    }
    return "unknown-mismatch"
}

/** Translate only structured host Tool parts into native LiteRT roles.
 * Text-only assistant turns retain their original MODEL role.
 * Unexecuted/malformed tool parts return null instead of inventing a tool result.
 */
internal fun nativeReplayHistory(
    history: List<Turn>,
    userMessage: (Turn) -> Message = { Message.user(it.rawText) },
): List<Message>? {
    val result = mutableListOf<Message>()
    for (turn in history) {
        when (turn.role) {
            ROLE_USER -> result.add(userMessage(turn))
            ROLE_ASSISTANT -> {
                if (turn.nativeParts.none { it is NativeReplayPart.Tool }) {
                    result.add(Message.model(turn.rawText))
                    continue
                }
                val pendingText = StringBuilder()
                var i = 0
                while (i < turn.nativeParts.size) {
                    when (val part = turn.nativeParts[i]) {
                        is NativeReplayPart.Text -> {
                            pendingText.append(part.text)
                            i++
                        }
                        is NativeReplayPart.Tool -> {
                            val calls = mutableListOf<ToolCall>()
                            val responses = mutableListOf<Content>()
                            // Adjacent tool parts belong to the same MODEL tool-call batch.
                            while (i < turn.nativeParts.size) {
                                val tool = turn.nativeParts[i] as? NativeReplayPart.Tool ?: break
                                val args = runCatching {
                                    Json.parseToJsonElement(tool.argumentsJson).let { it as? JsonObject }
                                }.getOrNull() ?: return null
                                if (tool.name.isBlank() || tool.output.isNullOrBlank()) return null
                                calls.add(ToolCall(tool.name, args.mapValues { (_, v) -> v.toToolValue() }))
                                responses.add(Content.ToolResponse(tool.name, tool.output))
                                i++
                            }
                            val textContents = if (pendingText.isEmpty()) {
                                Contents.of(emptyList<Content>())
                            } else Contents.of(pendingText.toString())
                            result.add(Message.model(contents = textContents, toolCalls = calls))
                            result.add(Message.tool(Contents.of(responses)))
                            pendingText.clear()
                        }
                    }
                }
                if (pendingText.isNotEmpty()) result.add(Message.model(pendingText.toString()))
            }
            else -> return null
        }
    }
    return result
}

private fun JsonElement.toToolValue(): Any? = when (this) {
    JsonNull -> null
    is JsonObject -> mapValues { (_, value) -> value.toToolValue() }
    is JsonArray -> map { it.toToolValue() }
    is JsonPrimitive -> if (isString) content else booleanOrNull ?: longOrNull ?: doubleOrNull ?: content
}

/** Rebuild a TOOL-ending assistant turn with real MODEL(toolCalls) -> TOOL roles.
 * The final TOOL message must be sent through sendMessageAsync, not prefilled twice.
 */
internal fun nativeColdToolContinuation(
    history: List<Turn>,
    userMessage: (Turn) -> Message = { Message.user(it.rawText) },
): Pair<List<Message>, Message>? {
    if (history.lastOrNull()?.role != ROLE_ASSISTANT) return null
    val messages = nativeReplayHistory(history, userMessage) ?: return null
    if (messages.size < 2 || messages.last().role != Role.TOOL ||
        messages[messages.lastIndex - 1].toolCalls.isEmpty()
    ) return null
    return messages.dropLast(1) to messages.last()
}

/** Use the public 0.19.1 TOOL-role API; no hand-written <tool_result> wrappers. */
internal fun nativeToolResponseMessage(results: List<RuntimeToolResult>): Message = Message.tool(
    Contents.of(results.map { Content.ToolResponse(it.name, it.output) })
)

/**
 * What [LiteRtRuntime.streamTurns] emits.
 *
 * The runtime disables the SDK's automatic tool calling, so a turn can end either with
 * generated text, with tool calls the host must execute, or with both. Tool calls are
 * surfaced rather than run here so the host can apply approval and the HARDLINE floor.
 */
sealed interface StreamEvent {
    /** The **cumulative** response assembled from SDK message chunks, not a delta.
     *  Downstream computes deltas itself. */
    data class Delta(val cumulative: String) : StreamEvent

    data class ThinkingDelta(val text: String) : StreamEvent

    /** Tool calls the model emitted this turn, in the order the model produced them. */
    data class ToolCalls(val calls: List<RuntimeToolCall>) : StreamEvent

    /** SDK-reported KV occupancy after a successful generation, never a delta. */
    data class ContextTokens(val count: Int) : StreamEvent
}

/**
 * Outcome of [planTurns]: whether the live [Conversation]'s KV cache can be reused.
 */
sealed interface TurnPlan {
    /** Reuse the warm Conversation; send only `history[sendFromIndex]` (guaranteed exactly
     *  one new turn). The KV cache already holds every turn before [sendFromIndex]. */
    data class Warm(val sendFromIndex: Int) : TurnPlan

    /** Recreate the Conversation and replay roles with initialMessages (or tool fallback). */
    data object Cold : TurnPlan
}

/**
 * Thrown when a `.litertlm` model file is structurally broken in a way that retrying
 * with a different accelerator might not fix (e.g. broken archive or schema mismatch).
 * The caller must NOT delete the model file automatically: JNI diagnostics can be misleading.
 */
class LiteRtModelCorruptException(
    val modelPath: String,
    cause: Throwable,
) : RuntimeException(
    "LiteRT model file appears corrupt or incompatible: ${cause.message}",
    cause,
)

/**
 * Thrown when engine init fails specifically inside the vision executor — the model file
 * itself is fine, but this device's GPU vision pipeline cannot be initialised. The
 * canonical signature is an error message containing `vision_litert_compiled_model_executor`
 * (the SDK's per-modality executor that fails when `CreateSharedMemoryManager` is
 * unimplemented in the OpenGL fallback path — upstream LiteRT-LM #2292, hitting Adreno
 * 7xx + OEM ROMs that block `libvndksupport.so` discovery).
 *
 * The runtime self-recovers by retrying with `supportImage=false` (text-only) once. The
 * provider catches this and persists the per-model decision so subsequent loads skip
 * the doomed GPU vision attempt entirely. Multimodal models still load — they just
 * cannot accept images on this device.
 */
class LiteRtVisionUnavailableException(
    val modelPath: String,
    cause: Throwable,
) : RuntimeException(
    "LiteRT vision encoder failed to initialise on this device's GPU: ${cause.message}",
    cause,
)

/**
 * Wraps Google's LiteRT-LM runtime (`com.google.ai.edge.litertlm:litertlm-android`, version
 * pinned in the version catalog) for on-device inference of `.litertlm` model files.
 *
 * # Why this rewrite (vs. the simpler v22A original)
 *
 * The original implementation made three SDK-misuse mistakes Gallery's
 * `LlmChatModelHelper` does NOT make. Each was silently broken on production devices:
 *
 *   1. **Skipped `engine.initialize()`.** `Engine(config)` returns a partially-constructed
 *      handle; only `initialize()` actually loads tokenizer, KV cache, and weights into the
 *      backend.
 *   2. **Did not pass `EngineConfig.maxNumTokens`.** The SDK falls back to an internal default
 *      (~16) when `maxNumTokens` is null, so users saw "I" then nothing.
 *   3. **Inlined the system prompt as `User: …\nAssistant: …` plain text.** The LiteRT-LM
 *      runtime ships per-model chat templates; without `systemInstruction` going through the
 *      template engine, Qwen2.5 (and any chat-tuned model) emits gibberish.
 *
 * # KV-cache reuse across turns (the perf rewrite)
 *
 * The original recreated the [Conversation] on every [ensureLoaded] call, so every turn
 * re-prefilled the ENTIRE conversation history from a cold KV cache — turn N paid for
 * turns 1..N every time. Gallery instead keeps ONE Conversation alive and sends only the
 * new user message each turn, so the KV cache stays warm and each turn prefills only its
 * own new tokens.
 *
 * This runtime now does the same. The Conversation is kept across turns when the
 * (model, accelerator, sampler, system instruction) tuple is unchanged. [streamTurns]
 * compares the caller's full turn list against [LoadedModel.processed] (the signatures the
 * live Conversation has already consumed): a clean single-turn append reuses the warm KV
 * cache. A completed native TOOL turn can also continue Warm, but only after its final
 * assistant rendering matches RikkaHub's returned history exactly. Edits, trimming,
 * cancellation and config changes still force Cold recovery via SDK initialMessages.
 * Non-user-ending tool continuations keep the compatibility text fallback.
 *
 * # Concurrency
 *
 * A single [mutex] serialises all access. The mutex is held for the full duration of each
 * inference, so [LoadedModel.processed] is only ever touched by one coroutine at a time.
 */
class LiteRtRuntime(
    private val context: Context,
    private val prefs: LocalRuntimePreferences,
) {

    /** The last successfully initialized normal-chat Engine, not the selected preference.
     * Null means no chat Engine is resident (including while Benchmark is running).
     * A GPU backend here denotes the configured LiteRT main backend, not every operator. */
    data class BackendStatus(
        val modelName: String,
        val backend: String,
        val preloadedOnly: Boolean,
        val note: String? = null,
    )

    private val _backendStatus = MutableStateFlow<BackendStatus?>(null)
    val backendStatus: StateFlow<BackendStatus?> = _backendStatus.asStateFlow()

    private fun actualBackendName(accel: String): String = when (accel) {
        "GPU" -> "GPU"
        "QNN", "NPU", "TPU" -> "NPU"
        else -> "CPU" // The existing NNAPI/unknown mapping uses Backend.CPU().
    }

    /** Call while holding [mutex] whenever the active Engine changes. */
    private fun publishBackendStatus() {
        val running = loaded
        _backendStatus.value = if (running != null) {
            BackendStatus(
                modelName = File(running.engineKey.modelPath).name,
                backend = actualBackendName(running.engineKey.accelerator),
                preloadedOnly = false,
                note = running.backendNote,
            )
        } else {
            preloaded?.let { staged ->
                BackendStatus(
                    modelName = File(staged.engineKey.modelPath).name,
                    backend = actualBackendName(staged.engineKey.accelerator),
                    preloadedOnly = true,
                    note = staged.backendNote,
                )
            }
        }
    }

    private val mutex = Mutex()
    private val generationMutex = Mutex()

    /** Reserve the full provider request, including the Engine-to-stream handoff. */
    suspend fun <T> withGenerationReservation(block: suspend () -> T): T = generationMutex.withLock { block() }
    private var loaded: LoadedModel? = null

    /** Engine-only prefetch; no Conversation/KV state until a real request arrives. */
    private var preloaded: PreloadedEngine? = null

    // Box-style backend selection is per model and explicit: no inferred CPU fallback,
    // process-level backend cache, or persistent native-crash blacklisting.

    // ---- Idle teardown ----
    //
    // The Engine + Conversation pin ~2-4 GB of native heap for the loaded model.
    // Keeping them warm is great for response latency on consecutive turns but expensive
    // for an app the user has wandered away from. After [idleTeardownDelayMs] of no
    // [streamTurns] activity, close the engine. Next request re-loads cold (10-60 s on
    // CPU). Configurable per-runtime via [setIdleTeardownDelayMs]; pass 0 to disable.

    @Volatile
    private var idleTeardownDelayMs: Long = TimeUnit.MINUTES.toMillis(5)

    @Volatile
    private var idleTeardownJob: kotlinx.coroutines.Job? = null

    private val idleScope = kotlinx.coroutines.CoroutineScope(
        kotlinx.coroutines.SupervisorJob() + Dispatchers.IO
    )

    init {
        // Restore the user's retention setting even if the app was restarted before any
        // settings page was opened. Changes made via Apply are observed automatically.
        idleScope.launch {
            prefs.liteRtRuntimeSettingsFlow(LocalRuntime.LiteRT)
                .map { it.idleTeardownMinutes }
                .distinctUntilChanged()
                .collect { minutes ->
                    mutex.withLock {
                        setIdleTeardownDelayMs(TimeUnit.MINUTES.toMillis(minutes.toLong()))
                    }
                }
        }
    }

    /** Set the idle window. 0 disables the watchdog (engine stays warm until manually
     *  closed). Negative values are clamped to 0. Honored on the next idle event. */
    fun setIdleTeardownDelayMs(ms: Long) {
        idleTeardownDelayMs = ms.coerceAtLeast(0L)
        // If a teardown was scheduled with the old delay, re-arm with the new one.
        armIdleTeardown()
    }

    /** Cancel any pending teardown and schedule a fresh one at the current delay. Called
     *  from the streamTurns finally so a new turn always resets the clock. */
    private fun armIdleTeardown() {
        idleTeardownJob?.cancel()
        val delay = idleTeardownDelayMs
        if (delay <= 0L || (loaded == null && preloaded == null)) return
        idleTeardownJob = idleScope.launch {
            kotlinx.coroutines.delay(delay)
            mutex.withLock {
                val now = android.os.SystemClock.elapsedRealtime()
                loaded?.let { current ->
                    if (current.pendingStreamOwner == null && now - current.lastUseAtMs >= delay) {
                        try { current.conversation.close() } catch (_: Throwable) {}
                        try { current.engine.close() } catch (_: Throwable) {}
                        loaded = null
                    }
                }
                preloaded?.let { staged ->
                    if (now - staged.lastUseAtMs >= delay) {
                        try { staged.engine.close() } catch (_: Throwable) {}
                        preloaded = null
                    }
                }
                publishBackendStatus()
            }
        }
    }

    /**
     * Snapshot of every config bit that requires Engine teardown. The Engine is the
     * expensive resource (cold-load 10-60s); changing any of these forces a full reload.
     */
    private data class EngineKey(
        val modelPath: String,
        val accelerator: String,
        val maxNumTokens: Int,
        val supportImage: Boolean,
        val supportAudio: Boolean,
        val speculativeDecoding: Boolean,
        /** Per-model vision-backend choice. Mirrors Gallery's `ConfigKeys.VISION_ACCELERATOR`
         *  field. "gpu" / "cpu" / "npu". Default "gpu" matches Gallery's default; null
         *  is reserved for non-multimodal models where supportImage is also false. */
        val visionAccelerator: String,
    )

    /**
     * Snapshot of every config bit that requires only Conversation recreation (cheap) — NOT
     * an Engine reload. When this is unchanged the warm Conversation (and its KV cache) is
     * kept across [ensureLoaded] calls.
     */
    private data class ConversationKey(
        val systemInstructionText: String?,
        val constrainedDecoding: Boolean,
        val topK: Int,
        val topP: Double,
        val temperature: Double,
        /** Exact, ordered declaration JSON strings; changes require a fresh Conversation. */
        val toolDeclarations: List<String>,
        /** Switching thinking mode invalidates the Conversation KV cache, not the Engine. */
        val thinkingConfig: ThinkingConfig?,
    )

    /** Everything needed to (re)build a [Conversation] for the cold path without re-deriving
     *  it from scattered call parameters. */
    private data class ConversationSpec(
        val systemInstruction: Contents?,
        val tools: List<ToolProvider>,
        val thinkingConfig: ThinkingConfig?,
        val constrainedDecoding: Boolean,
        val topK: Int,
        val topP: Double,
        val temperature: Double,
    )

    private class LoadedModel(
        val engineKey: EngineKey,
        var conversationKey: ConversationKey,
        var conversationSpec: ConversationSpec,
        val engine: Engine,
        var conversation: Conversation,
        /** Captured from `Capabilities.hasSpeculativeDecodingSupport()` at engine build
         *  time. Persisted so we can surface it on warm-reuse paths too. */
        val fileSupportsSpeculativeDecoding: Boolean,
        /** True iff [engineKey.speculativeDecoding] was true at engine.initialize() time.
         *  Surfaced by [LoadOutcome.speculativeDecodingEngaged]. */
        val speculativeDecodingEngaged: Boolean,
        val backendNote: String? = null,
        /** Signatures of the turns the live Conversation's KV cache currently holds, in
         *  order — the caller-supplied history turns plus one synthetic assistant turn for
         *  the response generated last. Cleared whenever the Conversation is recreated. */
        val processed: MutableList<String> = mutableListOf(),
        /** True only for a newly created Conversation that has never received tokens.
         *  An error/cancel makes this false even if [processed] is empty. */
        var pristineConversation: Boolean = true,
        /** Prefix already installed in a pristine Conversation via initialMessages. */
        var initialHistorySignatures: List<String> = emptyList(),
        /** Non-null only after the SDK surfaced tool calls and the host has yet to return results. */
        var pendingToolBatch: PendingToolBatch? = null,
        /** Await exact host-history reconciliation before allowing a post-tool Warm append. */
        var completedToolTurn: CompletedNativeToolTurn? = null,
        /** Set after ensureLoaded until streamTurns begins: protects that gap from a
         *  model-selection preload closing the Engine before its actual request starts. */
        var pendingStreamOwner: String? = null,
        /** Monotonic timestamp of the last [streamTurns] invocation. The idle-teardown
         *  watchdog ([armIdleTeardown]) uses this to decide whether to close the engine. */
        @Volatile var lastUseAtMs: Long = android.os.SystemClock.elapsedRealtime(),
    )

    private class PreloadedEngine(
        val engineKey: EngineKey,
        val engine: Engine,
        val fileSupportsSpeculativeDecoding: Boolean,
        val speculativeDecodingEngaged: Boolean,
        val backendNote: String? = null,
        @Volatile var lastUseAtMs: Long = android.os.SystemClock.elapsedRealtime(),
    )

    /** Result of best-effort Engine preload; Deferred means an existing request owns
     *  the currently loaded Engine and the preloader should retry after it finishes. */
    enum class PreloadResult { READY, DEFERRED, FAILED }

    /**
     * Pre-initialize only the model Engine. A real request creates the Conversation with
     * its exact system instruction, sampler settings and native tool declarations.
     * Shares [mutex] with ensureLoaded/streamTurns to avoid duplicate initialization.
     * Preload errors are best-effort; normal generation will report them to the caller.
     */
    @OptIn(ExperimentalApi::class)
    suspend fun preloadEngine(
        modelPath: String,
        preferredAccel: String = "GPU",
        maxNumTokens: Int = 4096,
        supportImage: Boolean = false,
        supportAudio: Boolean = false,
        visionAccelerator: String = "gpu",
    ): PreloadResult = mutex.withLock {
        require(preferredAccel in setOf("CPU", "GPU", "NPU")) {
            "Unsupported LiteRT backend: $preferredAccel"
        }
        val accel = preferredAccel
        // Provider currently uses speculativeDecoding=false for all models.
        val key = EngineKey(
            modelPath = modelPath,
            accelerator = accel,
            maxNumTokens = maxNumTokens,
            supportImage = supportImage,
            supportAudio = supportAudio,
            speculativeDecoding = false,
            visionAccelerator = visionAccelerator,
        )
        val existing = loaded
        if (existing?.engineKey == key) return@withLock PreloadResult.READY
        // streamTurns holds this same mutex for the entire generation. Protect also
        // the gap between ensureLoaded and streamTurns (which are separate calls).
        if (existing != null && shouldDeferModelSwitch(existing.pendingStreamOwner != null)) {
            return@withLock PreloadResult.DEFERRED
        }
        // User selected another local model while A is idle. Close A before loading B,
        // rather than keeping two large models resident on an 8 GB phone.
        if (existing != null) {
            try { existing.conversation.close() } catch (_: Throwable) {}
            try { existing.engine.close() } catch (_: Throwable) {}
            loaded = null
        }
        preloaded?.let { staged ->
            if (staged.engineKey == key) {
                staged.lastUseAtMs = android.os.SystemClock.elapsedRealtime()
                armIdleTeardown()
                return@withLock PreloadResult.READY
            }
        }
        // Replace only an unused staged Engine; never interrupt a live Conversation.
        try { preloaded?.engine?.close() } catch (_: Throwable) {}
        preloaded = null
        publishBackendStatus()

        val fileSupportsSpec = try {
            ModelInfo.from(modelPath).use { info ->
                (info as? ModelInfo.Llm)?.hasSpeculativeDecodingSupport() == true
            }
        } catch (_: Throwable) { false }
        try {
            val engine = createEngineWithBackend(key, accel)
            preloaded = PreloadedEngine(
                engineKey = key,
                engine = engine,
                fileSupportsSpeculativeDecoding = fileSupportsSpec,
                speculativeDecodingEngaged = false,
            )
            publishBackendStatus()
            android.util.Log.i(
                "LiteRtRuntime",
                "preloadEngine: ready ${File(modelPath).name}, $accel, " +
                    "$maxNumTokens tokens (conversation deferred)",
            )
            armIdleTeardown()
            return@withLock PreloadResult.READY
        } catch (t: Throwable) {
            android.util.Log.w("LiteRtRuntime", "preloadEngine: $accel failed", t)
        }
        // Do not prevent the normal generation path from retrying and reporting errors.
        PreloadResult.FAILED
    }

    /** Called by the provider in finally. If argument preparation or media conversion
     *  failed before streamTurns started, do not leave the old Engine permanently reserved. */
    suspend fun releasePendingStream(modelPath: String, requestId: String) = mutex.withLock {
        loaded?.takeIf {
            it.engineKey.modelPath == modelPath && it.pendingStreamOwner == requestId
        }?.let {
            it.pendingStreamOwner = null
            it.lastUseAtMs = android.os.SystemClock.elapsedRealtime()
            armIdleTeardown()
        }
    }

    /** Release only an unused staged Engine, without touching an active Conversation. */
    suspend fun discardPreloadedEngine(modelPath: String? = null) = mutex.withLock {
        val staged = preloaded ?: return@withLock
        if (modelPath == null || staged.engineKey.modelPath == modelPath) {
            try { staged.engine.close() } catch (_: Throwable) {}
            preloaded = null
            publishBackendStatus()
        }
    }

    /** Classify native initialization errors without changing the selected backend. */
    private fun classifyEngineError(modelPath: String, t: Throwable): Throwable {
        // Walk the cause chain — the SDK's native exception is typically the innermost
        // cause; surrounding wrappers add their own (less specific) messages.
        val msg = generateSequence<Throwable>(t) { it.cause }
            .map { it.message.orEmpty() }
            .joinToString("\n")
        if (isVisionExecutorError(msg)) {
            return LiteRtVisionUnavailableException(modelPath, t)
        }
        val isCorrupt =
            msg.contains("Invalid magic number", ignoreCase = true) ||
            msg.contains("Failed to decompress", ignoreCase = true) ||
            msg.contains("No KV cache inputs found", ignoreCase = true) ||
            msg.contains("FAILED_PRECONDITION", ignoreCase = true) ||
            (msg.contains("INVALID_ARGUMENT", ignoreCase = true) &&
                (msg.contains("tokenizer", ignoreCase = true) ||
                 msg.contains("Section not found", ignoreCase = true) ||
                 msg.contains("Uncompressed size", ignoreCase = true)))
        return if (isCorrupt) LiteRtModelCorruptException(modelPath, t) else t
    }

    /**
     * Map our internal accelerator label → the SDK's [Backend] sealed-class instance.
     *
     * NPU/TPU both map to `Backend.NPU` with the app's native library dir, matching
     * Gallery's `LlmChatModelHelper.kt`: both labels share the QNN delegate loader path.
     */
    private fun acceleratorToBackend(accel: String): Backend = when (accel) {
        "QNN", "NPU", "TPU" -> Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
        "GPU" -> Backend.GPU()
        else -> Backend.CPU()
    }

    /**
     * Configure the engine + conversation for the next [streamTurns] call.
     *
     * Engine reuse: when every [EngineKey] field matches, the (expensive) Engine is kept.
     * Conversation reuse: when the [ConversationKey] ALSO matches, the warm Conversation —
     * and its KV cache — is kept untouched, so the next [streamTurns] can warm-continue.
     * When only the ConversationKey changed (system instruction / sampler), the Engine is
     * kept but the Conversation is recreated and [LoadedModel.processed] is cleared.
     *
     * If the selected backend fails, report the error without changing the backend.
     *
     * Returns the resolved accelerator label that was actually used.
     */
    @OptIn(ExperimentalApi::class) // ExperimentalFlags.* + Capabilities.hasSpeculativeDecodingSupport
    suspend fun ensureLoaded(
        modelPath: String,
        requestId: String,
        preferredAccel: String = "GPU",
        maxNumTokens: Int = 4096,
        supportImage: Boolean = false,
        supportAudio: Boolean = false,
        speculativeDecoding: Boolean = false,
        /** "gpu" | "cpu" | "npu". Defaults to "gpu" matching Gallery's
         *  DEFAULT_VISION_ACCELERATOR. Ignored when supportImage = false. */
        visionAccelerator: String = "gpu",
        systemInstructionText: String? = null,
        tools: List<ToolProvider> = emptyList(),
        toolDeclarations: List<String> = emptyList(),
        thinkingConfig: ThinkingConfig? = null,
        constrainedDecoding: Boolean = false,
        topK: Int = 64,
        topP: Double = 0.95,
        temperature: Double = 1.0,
        /** Current user-ending history; initialMessages are used only for a new Conversation. */
        initialTurnHistory: List<Turn> = emptyList(),
    ): LoadOutcome = mutex.withLock {
        require(preferredAccel in setOf("CPU", "GPU", "NPU")) {
            "Unsupported LiteRT backend: $preferredAccel"
        }
        val accel = preferredAccel
        // Only resolve historical media when a new Conversation is really needed.
        val nativeHistory by lazy { nativeInitialHistory(initialTurnHistory, ::nativeUserMessage) }
        val nativeInitialMessages by lazy { nativeHistory.orEmpty() }
        val nativeInitialSignatures by lazy {
            if (nativeHistory != null) initialTurnHistory.dropLast(1).map { it.signature } else emptyList()
        }

        // Probe the file for speculative-decoding support BEFORE building the engine.
        val supportsSpeculativeDecoding = try {
            ModelInfo.from(modelPath).use { info ->
                (info as? ModelInfo.Llm)?.hasSpeculativeDecodingSupport() == true
            }
        } catch (_: Throwable) {
            false
        }
        val effectiveSpeculativeDecoding = speculativeDecoding && supportsSpeculativeDecoding

        val desiredEngineKey = EngineKey(
            modelPath = modelPath,
            accelerator = accel,
            maxNumTokens = maxNumTokens,
            supportImage = supportImage,
            supportAudio = supportAudio,
            speculativeDecoding = effectiveSpeculativeDecoding,
            visionAccelerator = visionAccelerator,
        )
        val systemInstruction: Contents? =
            if (!systemInstructionText.isNullOrBlank()) Contents.of(systemInstructionText) else null
        val desiredConversationKey = ConversationKey(
            systemInstructionText = systemInstructionText?.takeIf { it.isNotBlank() },
            constrainedDecoding = constrainedDecoding,
            topK = topK,
            topP = topP,
            temperature = temperature,
            toolDeclarations = toolDeclarations.toList(),
            thinkingConfig = thinkingConfig,
        )
        val desiredConversationSpec = ConversationSpec(
            systemInstruction = systemInstruction,
            tools = tools,
            thinkingConfig = thinkingConfig,
            constrainedDecoding = constrainedDecoding,
            topK = topK,
            topP = topP,
            temperature = temperature,
        )

        val current = loaded
        if (current != null && current.engineKey == desiredEngineKey) {
            if (current.conversationKey == desiredConversationKey) {
                // Full reuse — Engine AND Conversation kept. The KV cache (and therefore
                // [processed]) is left intact so the next streamTurns can warm-continue.
                current.pendingStreamOwner = requestId
                return@withLock LoadOutcome(
                    accelerator = accel,
                    visionEnabled = current.engineKey.supportImage,
                    visionFellBackToTextOnly = false,
                    fileSupportsSpeculativeDecoding = current.fileSupportsSpeculativeDecoding,
                    speculativeDecodingEngaged = current.speculativeDecodingEngaged,
                )
            }
            // Engine kept, Conversation config changed — recreate just the Conversation.
            // The KV cache is gone, so [processed] must be cleared.
            try { current.conversation.close() } catch (_: Throwable) {}
            current.conversation = try {
                createConversationWithFlags(
                    engine = current.engine,
                    backend = acceleratorToBackend(current.engineKey.accelerator),
                    spec = desiredConversationSpec,
                    initialMessages = nativeInitialMessages,
                )
            } catch (t: Throwable) {
                try { current.engine.close() } catch (_: Throwable) {}
                loaded = null
                publishBackendStatus()
                throw t
            }
            current.initialHistorySignatures = nativeInitialSignatures
            current.conversationKey = desiredConversationKey
            current.conversationSpec = desiredConversationSpec
            current.processed.clear()
            current.pendingToolBatch = null
            current.completedToolTurn = null
            current.pristineConversation = true
            current.pendingStreamOwner = requestId
            return@withLock LoadOutcome(
                accelerator = accel,
                visionEnabled = current.engineKey.supportImage,
                visionFellBackToTextOnly = false,
                fileSupportsSpeculativeDecoding = current.fileSupportsSpeculativeDecoding,
                speculativeDecodingEngaged = current.speculativeDecodingEngaged,
            )
        }

        // Reuse an Engine initialized while the user was opening/selecting this model.
        // The request's actual tools, system instruction and sampling config are known
        // only now, so create the Conversation here, never during preload.
        preloaded?.let { staged ->
            if (staged.engineKey == desiredEngineKey) {
                preloaded = null
                try {
                    val conv = createConversationWithFlags(
                        staged.engine,
                        acceleratorToBackend(staged.engineKey.accelerator),
                        desiredConversationSpec,
                        initialMessages = nativeInitialMessages,
                    )
                    loaded = LoadedModel(
                        engineKey = staged.engineKey,
                        conversationKey = desiredConversationKey,
                        conversationSpec = desiredConversationSpec,
                        engine = staged.engine,
                        conversation = conv,
                        fileSupportsSpeculativeDecoding = staged.fileSupportsSpeculativeDecoding,
                        speculativeDecodingEngaged = staged.speculativeDecodingEngaged,
                        backendNote = staged.backendNote,
                        initialHistorySignatures = nativeInitialSignatures,
                        pendingStreamOwner = requestId,
                    )
                    publishBackendStatus()
                    android.util.Log.i("LiteRtRuntime", "ensureLoaded: reused preloaded Engine")
                    armIdleTeardown()
                    return@withLock LoadOutcome(
                        accelerator = staged.engineKey.accelerator,
                        visionEnabled = staged.engineKey.supportImage,
                        visionFellBackToTextOnly = false,
                        fileSupportsSpeculativeDecoding = staged.fileSupportsSpeculativeDecoding,
                        speculativeDecodingEngaged = staged.speculativeDecodingEngaged,
                    )
                } catch (t: Throwable) {
                    try { staged.engine.close() } catch (_: Throwable) {}
                    publishBackendStatus()
                    android.util.Log.w("LiteRtRuntime", "preloaded Conversation failed; retrying the same backend", t)
                }
            } else {
                // Settings/model changed during the background load; discard stale Engine.
                try { staged.engine.close() } catch (_: Throwable) {}
                preloaded = null
                publishBackendStatus()
            }
        }

        // Engine swap path: tear down any prior Engine + Conversation.
        try { current?.conversation?.close() } catch (_: Throwable) {}
        try { current?.engine?.close() } catch (_: Throwable) {}
        loaded = null
        publishBackendStatus()

        // Match Box: load ONLY the backend explicitly selected for this model.
        // A failure is propagated to the provider; never retry on CPU or persist CPU.
        val loadResult = try {
            tryLoadWithBackend(
                desiredEngineKey, accel, desiredConversationSpec, nativeInitialMessages,
            )
        } catch (t: Throwable) {
            throw classifyEngineError(modelPath, t)
        }

        loaded = LoadedModel(
            engineKey = desiredEngineKey.copy(
                accelerator = loadResult.accelerator,
                supportImage = loadResult.supportImage,
            ),
            conversationKey = desiredConversationKey,
            conversationSpec = desiredConversationSpec,
            engine = loadResult.engine,
            conversation = loadResult.conversation,
            fileSupportsSpeculativeDecoding = supportsSpeculativeDecoding,
            speculativeDecodingEngaged = effectiveSpeculativeDecoding,
            backendNote = null,
            initialHistorySignatures = nativeInitialSignatures,
            pendingStreamOwner = requestId,
        )
        publishBackendStatus()
        android.util.Log.i(
            "LiteRtRuntime",
            "ensureLoaded: accel=${loadResult.accelerator} visionEnabled=${loadResult.supportImage} " +
                "fileSupportsSpecDecoding=$supportsSpeculativeDecoding " +
                "specDecodingEngaged=$effectiveSpeculativeDecoding",
        )
        LoadOutcome(
            accelerator = loadResult.accelerator,
            visionEnabled = loadResult.supportImage,
            visionFellBackToTextOnly = false,
            fileSupportsSpeculativeDecoding = supportsSpeculativeDecoding,
            speculativeDecodingEngaged = effectiveSpeculativeDecoding,
        )
    }

    private class LoadResult(
        val engine: Engine,
        val conversation: Conversation,
        val accelerator: String,
        /** Matches the model's explicit vision setting; no text-only init fallback. */
        val supportImage: Boolean,
    )

    /**
     * Result returned by [ensureLoaded]: the selected accelerator and engine capabilities.
     * [visionFellBackToTextOnly] remains for caller compatibility but is always false in
     * Box-style mode: an initialization failure no longer changes backend or modalities.
     */
    data class LoadOutcome(
        val accelerator: String,
        val visionEnabled: Boolean,
        val visionFellBackToTextOnly: Boolean,
        /** True iff `Capabilities.hasSpeculativeDecodingSupport()` returned true for the
         *  model file. */
        val fileSupportsSpeculativeDecoding: Boolean,
        /** True iff the caller requested speculative decoding AND the file supports it
         *  AND the SDK's experimental flag was set during engine init. */
        val speculativeDecodingEngaged: Boolean,
    )

    /**
     * Build + initialize an Engine, then open a Conversation. Throws if either step fails.
     *
     * Mirrors Gallery's `LlmChatModelHelper.initialize()`:
     *   - EngineConfig with explicit `maxNumTokens`
     *   - `cacheDir` only when modelPath sits in `/data/local/tmp`
     *   - the `enableSpeculativeDecoding` flag dance around construct + initialize
     */
    @OptIn(ExperimentalApi::class)
    private suspend fun tryLoadWithBackend(
        engineKey: EngineKey,
        accel: String,
        conversationSpec: ConversationSpec,
        initialMessages: List<Message> = emptyList(),
    ): LoadResult {
        val engine = createEngineWithBackend(engineKey, accel)
        val conv = try {
            createConversationWithFlags(
                engine = engine,
                backend = acceleratorToBackend(accel),
                spec = conversationSpec,
                initialMessages = initialMessages,
            )
        } catch (t: Throwable) {
            try { engine.close() } catch (_: Throwable) {}
            throw t
        }
        return LoadResult(engine, conv, accel, engineKey.supportImage)
    }

    /** Engine initialization shared by cold generation and background preload. */
    @OptIn(ExperimentalApi::class)
    private suspend fun createEngineWithBackend(engineKey: EngineKey, accel: String): Engine {
        val backend = acceleratorToBackend(accel)
        // Vision backend MUST be GPU for Gemma 3n; audio backend MUST be CPU (Gallery's
        // mandate). Leave each null when the caller didn't request that modality so the
        // engine doesn't allocate the corresponding executor memory.
        // Vision backend mirrors Gallery's `LlmChatModelHelper`: configurable per model
        // via the visionAccelerator label. GPU is default (Gemma 3n / 4 train with GPU
        // vision); CPU and NPU are valid alternates the SDK accepts.
        val visionBackend: Backend? = if (engineKey.supportImage) {
            when (engineKey.visionAccelerator) {
                "cpu" -> Backend.CPU()
                "npu", "tpu" -> Backend.NPU(nativeLibraryDir = context.applicationInfo.nativeLibraryDir)
                else -> Backend.GPU()
            }
        } else null
        val audioBackend: Backend? = if (engineKey.supportAudio) Backend.CPU() else null

        // cacheDir: matches Google AI Edge Gallery's LlmChatModelHelper exactly. Only set
        // when the model path is the ADB push debug location (/data/local/tmp). For
        // installed models (filesDir/local-models/...) the SDK does its own cache
        // management internally; passing a non-null cacheDir for those paths diverges
        // from Gallery's reference build with unknown side effects on vision-backend init.
        val engineConfig = EngineConfig(
            modelPath = engineKey.modelPath,
            backend = backend,
            visionBackend = visionBackend,
            audioBackend = audioBackend,
            maxNumTokens = engineKey.maxNumTokens,
            cacheDir = if (engineKey.modelPath.startsWith("/data/local/tmp"))
                context.getExternalFilesDir(null)?.absolutePath
            else null,
        )

        val engine = withContext(Dispatchers.IO) {
            // Daily conversations should not enable LiteRT-LM benchmark telemetry.
            // Normal chat does not collect BenchmarkInfo. The standalone benchmark()
            // uses its own native handle under the same runtime mutex.
            ExperimentalFlags.enableBenchmark = false
            // The speculative-decoding flag is temporary and reset after initialize().
            ExperimentalFlags.enableSpeculativeDecoding = engineKey.speculativeDecoding
            val e = try {
                Engine(engineConfig).also { built ->
                    try {
                        val initStartedNs = System.nanoTime()
                        built.initialize()
                        android.util.Log.i(
                            "LiteRtRuntime",
                            "perf: engineInitMs=${(System.nanoTime() - initStartedNs) / 1_000_000L} " +
                                "accelerator=$accel model=${File(engineKey.modelPath).name}",
                        )
                    } catch (t: Throwable) {
                        // Close the partially-constructed engine to release its native
                        // handle before propagating.
                        try { built.close() } catch (_: Throwable) {}
                        throw t
                    }
                }
            } finally {
                ExperimentalFlags.enableSpeculativeDecoding = false
            }
            e
        }

        return engine
    }

    /**
     * Build a Conversation with the constrained-decoding flag dance.
     *
     * Sampler choice mirrors Gallery:
     *   - NPU backend → samplerConfig MUST be null.
     *   - GPU/CPU backend → explicit SamplerConfig with the caller's topK/topP/temperature.
     */
    @OptIn(ExperimentalApi::class)
    private fun createConversationWithFlags(
        engine: Engine,
        backend: Backend,
        spec: ConversationSpec,
        initialMessages: List<Message> = emptyList(),
    ): Conversation {
        ExperimentalFlags.enableConversationConstrainedDecoding = spec.constrainedDecoding
        return try {
            val startedNs = System.nanoTime()
            val conversation = engine.createConversation(
                ConversationConfig(
                    samplerConfig = if (backend is Backend.NPU) {
                        null
                    } else {
                        SamplerConfig(
                            topK = spec.topK,
                            topP = spec.topP,
                            temperature = spec.temperature,
                        )
                    },
                    systemInstruction = spec.systemInstruction,
                    initialMessages = initialMessages,
                    tools = spec.tools,
                    thinkingConfig = spec.thinkingConfig,
                    // Surface tool calls instead of dispatching them. The SDK would happily
                    // run them itself, but that path skips the host's approval prompt, the
                    // auto-approve allowlist, the HARDLINE floor, and the per-turn
                    // wall-clock budget. [LiteRtProvider] republishes the calls as ordinary
                    // tool parts so the normal execution path applies.
                    automaticToolCalling = false,
                )
            )
            android.util.Log.i(
                "LiteRtRuntime",
                "perf: conversationInitMs=${(System.nanoTime() - startedNs) / 1_000_000L} " +
                    "backend=${backend.name}",
            )
            conversation
        } finally {
            ExperimentalFlags.enableConversationConstrainedDecoding = false
        }
    }

    /** Close + rebuild the live Conversation in place, clearing its KV cache. Caller must
     *  hold [mutex] and must clear [LoadedModel.processed] itself. */
    private fun recreateConversationLocked(
        instance: LoadedModel,
        initialMessages: List<Message> = emptyList(),
    ) {
        try { instance.conversation.close() } catch (_: Throwable) {}
        instance.conversation = try {
            createConversationWithFlags(
                engine = instance.engine,
                backend = acceleratorToBackend(instance.engineKey.accelerator),
                spec = instance.conversationSpec,
                initialMessages = initialMessages,
            )
        } catch (t: Throwable) {
            try { instance.engine.close() } catch (_: Throwable) {}
            loaded = null
            publishBackendStatus()
            throw t
        }
        instance.pristineConversation = true
        instance.pendingToolBatch = null
        instance.completedToolTurn = null
        instance.initialHistorySignatures = emptyList() // overwritten by streamTurns for native cold
    }

    /**
     * Stream one assistant response.
     *
     * [history] is the FULL turn list for the conversation (already trimmed to the model's
     * context budget by the caller). The runtime decides — via [planTurns] — whether the
     * live Conversation's KV cache already holds a prefix of [history]:
     *
     *  - **Warm:** the cache holds everything except exactly one newly-appended turn → send
     *    only that turn's [Turn.rawText]; the SDK's chat template applies role wrapping and
     *    the prior turns are reused from the cache (Gallery's behaviour).
     *  - **Native TOOL:** execute only after AgentLoop approval, submit SDK TOOL-role
     *    results, and reconcile the completed assistant against returned host history
     *    before the next ordinary Warm append.
     *  - **Cold:** first turn, changed history or unsupported input → recreate with
     *    role-bearing initialMessages when the last turn is USER. [coldBlob] remains
     *    the compatibility path for unmatched tool continuations ending ASSISTANT.
     *
     * Each emitted String is the **cumulative** response so far, NOT a delta — same contract
     * as Gallery's `partialResult`. Downstream consumers compute deltas themselves.
     *
     * Caller MUST have called [ensureLoaded] first. The [mutex] is held for the whole
     * inference so two concurrent callers queue up rather than racing the Conversation.
     */
    @OptIn(ExperimentalApi::class)
    fun streamTurns(
        requestId: String,
        history: List<Turn>,
        coldBlob: String,
        toolResults: List<RuntimeToolResult> = emptyList(),
        images: List<Bitmap> = emptyList(),
        audioClips: List<ByteArray> = emptyList(),
        onThinking: ((String) -> Unit)? = null,
        maxOutputToken: Int? = null,
    ): Flow<StreamEvent> = channelFlow {
        // GPU-boost the inference. Three levers, in order of effectiveness:
        //   1. PerformanceHintManager (API 33+). Opens a hint session for this thread
        //      with a 50ms target — the OS scheduler treats the process as doing
        //      sustained compute and refuses to downclock GPU/CPU like it would for a
        //      productivity app. This is exactly what Android game engines use.
        //   2. THREAD_PRIORITY_URGENT_DISPLAY on the calling thread. LiteRT's native
        //      worker threads inherit nice values from their creator; boosting our
        //      thread propagates a higher scheduling priority into the compute work.
        //   3. (Manifest) game_mode_config.xml declares we accept PERFORMANCE mode,
        //      so OEM game-mode frameworks (Adreno GPP, Mali frame rate control,
        //      Tensor Game Mode) also bump GPU clocks + relax DCVS throttling.
        // All cleaned up in the finally so an idle app doesn't keep the boost.
        val callerTid = Process.myTid()
        val originalPriority = runCatching {
            Process.getThreadPriority(callerTid)
        }.getOrDefault(Process.THREAD_PRIORITY_DEFAULT)
        val hintSession = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            runCatching {
                val phm = context.getSystemService(PerformanceHintManager::class.java)
                phm?.createHintSession(intArrayOf(callerTid), 50_000_000L)  // 50 ms target
            }.getOrNull()
        } else null
        runCatching {
            Process.setThreadPriority(callerTid, Process.THREAD_PRIORITY_URGENT_DISPLAY)
        }

        try { mutex.withLock {
            val instance = loaded
                ?: throw IllegalStateException("Call ensureLoaded(...) before streamTurns()")

            val hasMedia = images.isNotEmpty() || audioClips.isNotEmpty()
            val historySignatures = history.map { it.signature }
            // The provider has now entered its generation call, so a pending-request
            // reservation is no longer needed; the mutex protects the active stream.
            check(instance.pendingStreamOwner == requestId) {
                "LiteRT request was superseded before generation started; retry this turn"
            }
            instance.pendingStreamOwner = null
            // A native TOOL continuation completed on the previous request. Only now
            // can we see RikkaHub's finalized assistant part ordering and transformations.
            // Reconcile against actual UI history before trusting the SDK's warm KV state.
            val completedToolTurn = instance.completedToolTurn
            val syncedToolPrefix = reconciledToolTurnSignatures(completedToolTurn, history)
            instance.completedToolTurn = null  // one-shot; mismatches must fall back to Cold
            if (syncedToolPrefix != null) {
                instance.processed.clear()
                instance.processed.addAll(syncedToolPrefix)
                android.util.Log.i("LiteRtRuntime", "streamTurns: tool turn history synchronized")
            } else if (completedToolTurn != null) {
                // No user content in logs; report only the safe Cold decision.
                android.util.Log.d("LiteRtRuntime", "streamTurns: tool turn history mismatch; Cold recovery")
            }
            // The host has finished approval/execution and returned the tool outputs.
            // Continue the SAME Conversation with a real TOOL-role message only when the
            // exact pending call ids, arguments and unchanged conversation prefix match.
            // Otherwise use the existing Cold recovery path (edits/retries/reloads).
            val nativeToolResults = if (!hasMedia) {
                matchNativeToolResults(instance.pendingToolBatch, history, toolResults)
            } else null
            if (toolResults.isNotEmpty() && nativeToolResults == null) {
                android.util.Log.w(
                    "LiteRtRuntime",
                    "streamTurns: native TOOL continuation unavailable: " +
                        nativeToolMatchFailureReason(instance.pendingToolBatch, history, toolResults) +
                        "; hostTurns=${history.size}; results=${toolResults.size}",
                )
            }
            // Construct a role-correct Cold fallback only for actual executed tool parts.
            // Never parse tool-looking text generated by the model into executable calls.
            val coldToolReplay = if (nativeToolResults == null && !hasMedia && toolResults.isNotEmpty()) {
                nativeColdToolContinuation(history, ::nativeUserMessage)
            } else null
            val planned = planTurns(instance.processed, historySignatures, hasMedia)
            // Only a newly appended USER can be submitted as a user message.
            val plan = if (planned is TurnPlan.Warm &&
                history[planned.sendFromIndex].role != ROLE_USER
            ) TurnPlan.Cold else planned

            val inputText: String = if (nativeToolResults != null) {
                android.util.Log.i(
                    "LiteRtRuntime",
                    "streamTurns: native TOOL continuation, responses=${nativeToolResults.size}",
                )
                ""
            } else when (plan) {
                is TurnPlan.Warm -> {
                    android.util.Log.d("LiteRtRuntime", "streamTurns: Warm append")
                    history[plan.sendFromIndex].rawText
                }
                TurnPlan.Cold -> {
                    // Both normal Cold and tool-result Cold use real SDK roles.
                    // The compatibility text blob remains for unsupported non-tool agent turns.
                    val expectedSignatures = if (history.lastOrNull()?.role == ROLE_USER) {
                        history.dropLast(1).map { it.signature }
                    } else emptyList()
                    // Reuse a pristine matching native prefix without re-decoding attachments.
                    val canReusePristine = coldToolReplay == null && history.lastOrNull()?.role == ROLE_USER &&
                        instance.pristineConversation && instance.initialHistorySignatures == expectedSignatures
                    val nativeHistory = if (canReusePristine) emptyList() else
                        coldToolReplay?.first ?: nativeInitialHistory(history, ::nativeUserMessage)
                    val nativeSignatures = if (nativeHistory != null && coldToolReplay == null) expectedSignatures else emptyList()
                    if (!canReusePristine) {
                        recreateConversationLocked(instance, nativeHistory.orEmpty())
                        instance.initialHistorySignatures = nativeSignatures
                        android.util.Log.i(
                            "LiteRtRuntime",
                            "streamTurns: Cold recreated; nativeHistory=${nativeHistory?.size ?: -1}",
                        )
                    } else {
                        android.util.Log.d("LiteRtRuntime", "streamTurns: Cold pristine Conversation with matching prefix")
                    }
                    instance.processed.clear()
                    if (coldToolReplay != null) {
                        android.util.Log.i(
                            "LiteRtRuntime",
                            "streamTurns: Cold native TOOL replay; initialMessages=${nativeHistory?.size ?: 0}",
                        )
                    } else if (nativeHistory == null) {
                        android.util.Log.w("LiteRtRuntime", "streamTurns: unsupported Cold fallback to plain text history")
                    }
                    if (coldToolReplay != null) "" else if (nativeHistory != null) history.last().rawText else coldBlob
                }
            }

            val conv = instance.conversation
            // The other place a broken GPU delegate faults: it can survive engine init and
            // only die once real work is submitted. Re-arm for every turn; the first token
            // below disarms.
            // Build Contents in Gallery's order: images and audio first, text last.
            val contentList = mutableListOf<Content>()
            for (image in images) contentList.add(Content.ImageBytes(image.toPngByteArray()))
            for (clip in audioClips) contentList.add(Content.AudioBytes(clip))
            if (inputText.trim().isNotEmpty()) contentList.add(Content.Text(inputText))
            val inputMessage = when {
                nativeToolResults != null -> nativeToolResponseMessage(nativeToolResults)
                coldToolReplay != null -> coldToolReplay.second
                else -> Message.user(Contents.of(contentList))
            }
            // Consumed (or invalidated) before sending: a failed/cancelled call may not
            // replay the same results into a partially-mutated native Conversation.
            instance.pendingToolBatch = null

            // Set once the SDK reports the generation finished (either way). Read from the
            // flow's coroutine to tell a natural end from a cancellation.
            val generationFinished = java.util.concurrent.atomic.AtomicBoolean(false)
            val cancellationRequested = java.util.concurrent.atomic.AtomicBoolean(false)
            val nativeFinished = CompletableDeferred<Throwable?>()
            val responseBuilder = StringBuilder()
            // How many of the message's tool calls have already been forwarded. The SDK
            // re-delivers the full list on each callback, so forward only the new tail.
            var emittedToolCalls = 0
            val requestedTools = mutableListOf<RuntimeToolCall>()
            // From this point onward native state may have changed, even if a send
            // throws or the user cancels before the first token arrives.
            instance.pristineConversation = false
            conv.sendMessageAsync(
                inputMessage,
                object : MessageCallback {
                    override fun onMessage(message: Message) {
                        message.channels["thought"]?.let { thinking ->
                            if (thinking.isNotEmpty()) {
                                onThinking?.invoke(thinking)
                                trySendBlocking(StreamEvent.ThinkingDelta(thinking))
                            }
                        }
                        val text = message.toString()
                        if (text.isNotEmpty()) {
                            responseBuilder.append(text)
                            trySendBlocking(StreamEvent.Delta(responseBuilder.toString()))
                        }
                        val calls = message.toolCalls
                        if (calls.size > emittedToolCalls) {
                            val fresh = calls.drop(emittedToolCalls).mapIndexed { i, call ->
                                RuntimeToolCall(
                                    id = "litert-$requestId-tool-${emittedToolCalls + i}",
                                    name = call.name,
                                    argumentsJson = argumentsToJson(call.arguments),
                                )
                            }
                            emittedToolCalls = calls.size
                            requestedTools.addAll(fresh)
                            trySendBlocking(StreamEvent.ToolCalls(fresh))
                        }
                    }
                    override fun onDone() {
                        val fullResponse = responseBuilder.toString()
                        // The Conversation's KV cache now holds [history…] + the assistant
                        // turn just generated. Record that so the NEXT call can warm-continue
                        // if it is a clean single-turn append. Mutating [processed] here is
                        // safe: the mutex is held for this whole streamTurns invocation.
                        instance.processed.clear()
                        instance.completedToolTurn = null
                        if (!cancellationRequested.get()) {
                            if (nativeToolResults == null && coldToolReplay == null) {
                                // Ordinary turns are represented in RikkaHub as one newly
                                // appended assistant message; regular Warm behaviour is safe.
                                instance.processed.addAll(historySignatures)
                                instance.processed.add(turnSignature(ROLE_ASSISTANT, fullResponse))
                            } else if (requestedTools.isEmpty() && fullResponse.isNotBlank()) {
                                // SDK native history uses MODEL/TOOL/MODEL, while RikkaHub
                                // stores these in one ASSISTANT UIMessage. Reconcile its
                                // exact final rendering on the next call before Warm reuse.
                                instance.completedToolTurn = CompletedNativeToolTurn(
                                    precedingTurns = history.dropLast(1).toList(),
                                    assistantBeforeFinalReply = history.last().rawText,
                                    finalReply = fullResponse,
                                )
                            }
                            if (requestedTools.isNotEmpty()) {
                                // Another tool step must be approved/executed by AgentLoop;
                                // do not mark this multi-tool sequence as complete yet.
                                val anchor = if (history.lastOrNull()?.role == ROLE_ASSISTANT) {
                                    historySignatures.dropLast(1)
                                } else historySignatures
                                instance.pendingToolBatch = PendingToolBatch(anchor, requestedTools.toList())
                            }
                        }
                        instance.lastUseAtMs = android.os.SystemClock.elapsedRealtime()
                        generationFinished.set(true)
                        nativeFinished.complete(null)
                    }
                    override fun onError(throwable: Throwable) {
                        // The KV-cache state is now unknown — force the next turn cold.
                        instance.processed.clear()
                        instance.pendingToolBatch = null
                        instance.completedToolTurn = null
                        instance.lastUseAtMs = android.os.SystemClock.elapsedRealtime()
                        generationFinished.set(true)
                        nativeFinished.complete(throwable)
                    }
                },
                emptyMap(),
                maxOutputToken = maxOutputToken,
            )
            try {
                // Do not close the response channel in onDone: publish the final KV
                // snapshot from this coroutine, outside the native callback thread.
                val error = nativeFinished.await()
                if (error == null) {
                    val count = runCatching { conv.getTokenCount() }
                        .onFailure { android.util.Log.d("LiteRtRuntime", "getTokenCount unavailable", it) }
                        .getOrNull()
                    if (count != null && count >= 0) {
                        android.util.Log.i(
                            "LiteRtRuntime",
                            "streamTurns: kvTokens=$count/${instance.engineKey.maxNumTokens}",
                        )
                        send(StreamEvent.ContextTokens(count))
                    }
                } else if (error !is CancellationException) {
                    throw error
                }
            } finally {
                // A cancelled collector must request native stop before another turn
                // can use the Engine. Keep the mutex until acknowledgment or timeout.
                if (!generationFinished.get()) {
                    cancellationRequested.set(true)
                    android.util.Log.i("LiteRtRuntime", "stream cancelled; request native stop")
                    runCatching { conv.cancelProcess() }
                        .onFailure { android.util.Log.w("LiteRtRuntime", "cancelProcess failed", it) }
                    instance.processed.clear()
                    instance.pendingToolBatch = null
                    instance.completedToolTurn = null
                }
                if (cancellationRequested.get()) {
                    withContext(NonCancellable) {
                        val acknowledged = withTimeoutOrNull(1_500L) {
                            nativeFinished.await()
                            true
                        } ?: false
                        if (!acknowledged) {
                            android.util.Log.w(
                                "LiteRtRuntime",
                                "native cancellation not acknowledged within 1500ms; next turn must rebuild",
                            )
                        }
                    }
                }
            }
        } } finally {
            runCatching { hintSession?.close() }
            runCatching { Process.setThreadPriority(callerTid, originalPriority) }
            // Arm (or re-arm) the idle teardown. A new turn cancels the prior schedule and
            // starts a fresh window from now; no turn keeps the existing one ticking.
            armIdleTeardown()
        }
    }.flowOn(Dispatchers.IO)

    /**
     * Cancel the currently-running generation, if any. Safe to call when nothing is
     * generating. Does NOT tear down the Engine or Conversation; the runtime stays warm.
     */
    fun stop() {
        try { loaded?.conversation?.cancelProcess() } catch (_: Throwable) {}
    }

    /**
     * Serializes the standalone LiteRT-LM benchmark with normal chat and preloading.
     * Releasing the chat engine avoids two multi-GB native engines competing on 8 GB devices.
     * A request already generating holds [mutex] until it completes. New requests wait
     * until benchmark() returns. Never expose the benchmark engine as [loaded].
     *
     * JNI benchmark() is synchronous; coroutine cancellation is checked between runs.
     */
    @OptIn(ExperimentalApi::class)
    suspend fun <T> withExclusiveBenchmark(
        modelPath: String, accelerator: String, block: suspend () -> T,
    ): T {
        mutex.lock()
        try {
            // ensureLoaded -> streamTurns is a two-call handoff. Do not take its
            // engine away in that gap even though the mutex is briefly unlocked.
            check(loaded?.pendingStreamOwner == null) {
                "A local generation is about to start; retry Benchmark when it finishes"
            }
            idleTeardownJob?.cancel()
            idleTeardownJob = null
            try { loaded?.conversation?.close() } catch (_: Throwable) {}
            try { loaded?.engine?.close() } catch (_: Throwable) {}
            try { preloaded?.engine?.close() } catch (_: Throwable) {}
            loaded = null
            preloaded = null
            publishBackendStatus()
            // The normal chat Engine always starts with this flag off.
            ExperimentalFlags.enableBenchmark = false
            return block()
        } finally {
            mutex.unlock()
        }
    }

    /**
     * Tear down the currently loaded engine + conversation, if any.
     */
    suspend fun closeIfLoaded() = generationMutex.withLock {
        mutex.withLock {
            idleTeardownJob?.cancel()
            idleTeardownJob = null
            try { loaded?.conversation?.close() } catch (_: Throwable) {}
            try { loaded?.engine?.close() } catch (_: Throwable) {}
            try { preloaded?.engine?.close() } catch (_: Throwable) {}
            loaded = null
            preloaded = null
            publishBackendStatus()
        }
    }

    /** Reconstruct media from the original saved attachment when a Cold prefix is rebuilt. */
    private fun nativeUserMessage(turn: Turn): Message {
        val parts = buildList<Content> {
            for (image in turn.images) {
                val bitmap = image.toLiteRtBitmap(context)
                    ?: error("Cannot restore a historical LiteRT image attachment")
                try { add(Content.ImageBytes(bitmap.toPngByteArray())) } finally { bitmap.recycle() }
            }
            for (audio in turn.audio) {
                add(Content.AudioBytes(audio.toLiteRtAudioBytes(context)
                    ?: error("Cannot restore a historical LiteRT audio attachment")))
            }
            if (turn.rawText.isNotBlank()) add(Content.Text(turn.rawText))
        }
        return Message.user(Contents.of(parts))
    }

    private fun Bitmap.toPngByteArray(): ByteArray {
        val stream = ByteArrayOutputStream()
        this.compress(Bitmap.CompressFormat.PNG, 100, stream)
        return stream.toByteArray()
    }

    companion object {
        /**
         * Pure decision function: given the signatures the live Conversation has already
         * consumed ([processed]) and the caller's full [historySignatures], decide whether
         * the warm KV cache can be reused. Extracted for unit testing.
         *
         * Warm reuse requires ALL of:
         *  - no media inputs (the warm path sends one turn's text and has nowhere to attach
         *    per-call images/audio onto a prior turn);
         *  - the runtime has prior warm state ([processed] non-empty);
         *  - [processed] is a strict prefix of [historySignatures];
         *  - exactly one turn was appended (a clean continuation — more than one, or a
         *    rewritten earlier turn, goes cold so the full history is re-sent correctly).
         *
         * Any "no" answer is [TurnPlan.Cold], which is always correct — the warm path is
         * purely an optimisation.
         */
        /**
         * SDK-native Cold recovery. The newest user input must NOT be in initialMessages:
         * it is passed separately to sendMessageAsync and includes current-turn media.
         * Null means this is an agent/tool continuation, not a fresh user turn.
         */
        internal fun nativeInitialHistory(
            history: List<Turn>,
            userMessage: (Turn) -> Message = { Message.user(it.rawText) },
        ): List<Message>? {
            if (history.lastOrNull()?.role != ROLE_USER) return null
            return nativeReplayHistory(history.dropLast(1), userMessage)
        }

        internal fun shouldRecreateForCold(pristineConversation: Boolean): Boolean =
            !pristineConversation

        internal fun shouldDeferModelSwitch(pendingStream: Boolean): Boolean =
            pendingStream

        internal fun planTurns(
            processed: List<String>,
            historySignatures: List<String>,
            hasMedia: Boolean,
        ): TurnPlan {
            if (hasMedia) return TurnPlan.Cold
            if (processed.isEmpty()) return TurnPlan.Cold
            if (processed.size >= historySignatures.size) return TurnPlan.Cold
            if (historySignatures.subList(0, processed.size) != processed) return TurnPlan.Cold
            if (historySignatures.size - processed.size != 1) return TurnPlan.Cold
            return TurnPlan.Warm(sendFromIndex = processed.size)
        }

        /**
         * True if the joined cause-chain message text indicates the SDK's vision-modality
         * executor failed at compile / init time. Extracted as a pure function for unit
         * testing. The canonical signature is `vision_litert_compiled_model_executor` (the
         * source file path in the SDK's native stack), but in practice the error can also
         * surface as `vision_litert` truncated, `CreateSharedMemoryManager is not
         * implemented` (the upstream root cause on Adreno 7xx + One UI / OriginOS), or
         * `gpu_backend_opengl.cc` (the file that hosts the stub). Match any of those.
         */
        /**
         * Serialise a `ToolCall`'s argument map to a JSON object string, matching the
         * `input` field of `UIMessagePart.Tool` so the host can parse it exactly like a
         * cloud provider's arguments payload.
         *
         * The SDK decodes arguments into plain Kotlin values (`String`, `Number`,
         * `Boolean`, `List`, `Map`, null) and its own converter is internal, so we walk the
         * structure ourselves. An unrecognised leaf is stringified rather than dropped: a
         * tool receiving a stringly-typed argument can still fail loudly, whereas a missing
         * key looks like the model never supplied it.
         */
        internal fun argumentsToJson(arguments: Map<String, Any?>): String =
            JsonObject(arguments.mapValues { (_, v) -> toJsonElement(v) }).toString()

        private fun toJsonElement(value: Any?): JsonElement = when (value) {
            null -> JsonNull
            is String -> JsonPrimitive(value)
            is Boolean -> JsonPrimitive(value)
            is Number -> JsonPrimitive(value)
            is Map<*, *> -> JsonObject(
                value.entries.associate { (k, v) -> k.toString() to toJsonElement(v) }
            )

            is Iterable<*> -> JsonArray(value.map { toJsonElement(it) })
            is Array<*> -> JsonArray(value.map { toJsonElement(it) })
            else -> JsonPrimitive(value.toString())
        }

        internal fun isVisionExecutorError(joinedMessage: String): Boolean =
            joinedMessage.contains("vision_litert", ignoreCase = true) ||
                joinedMessage.contains("CreateSharedMemoryManager", ignoreCase = true) ||
                joinedMessage.contains("gpu_backend_opengl", ignoreCase = true)
    }
}
