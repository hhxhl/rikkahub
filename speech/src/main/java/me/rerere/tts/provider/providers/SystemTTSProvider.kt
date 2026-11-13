package me.rerere.tts.provider.providers

import android.content.Context
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeout
import me.rerere.common.android.appTempFolder
import me.rerere.tts.model.AudioChunk
import me.rerere.tts.model.AudioFormat
import me.rerere.tts.model.TTSRequest
import me.rerere.tts.provider.TTSProvider
import me.rerere.tts.provider.TTSProviderSetting
import java.io.File
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicReference

private const val INIT_TIMEOUT_MS = 15_000L
private const val SYNTH_TIMEOUT_MS = 120_000L
private const val VOICE_EXIT_RELEASE_MS = 60_000L

/** Reuses Android's TextToSpeech engine; ExoPlayer still handles actual audio playback. */
class SystemTTSProvider : TTSProvider<TTSProviderSetting.SystemTTS> {
    private data class PendingSynthesis(
        val id: String,
        val completed: CompletableDeferred<Unit> = CompletableDeferred()
    )

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val synthesisMutex = Mutex()
    private val pending = AtomicReference<PendingSynthesis?>(null)
    private val lifecycleLock = Any()

    // Only access engine while synthesisMutex is held.
    private var engine: TextToSpeech? = null
    // Protected by lifecycleLock so UI events and synthesis jobs cannot race.
    private var voiceModeActive = false
    private var releaseJob: Job? = null
    private var warmupJob: Job? = null

    private val listener = object : UtteranceProgressListener() {
        override fun onStart(utteranceId: String?) = Unit
        override fun onDone(utteranceId: String?) = finish(utteranceId, null)

        @Deprecated("Required for older Android TTS implementations")
        override fun onError(utteranceId: String?) =
            finish(utteranceId, IllegalStateException("System TTS synthesis failed"))

        override fun onError(utteranceId: String?, errorCode: Int) =
            finish(utteranceId, IllegalStateException("System TTS error: $errorCode"))

        override fun onStop(utteranceId: String?, interrupted: Boolean) =
            finish(utteranceId, IllegalStateException("System TTS synthesis stopped"))
    }

    private fun finish(id: String?, error: Throwable?) {
        val task = pending.get() ?: return
        if (id != task.id) return
        if (error == null) task.completed.complete(Unit)
        else task.completed.completeExceptionally(error)
    }

    /** Called by the active voice session, never during normal app launch. */
    fun setVoiceModeActive(context: Context, active: Boolean) {
        synchronized(lifecycleLock) {
            if (voiceModeActive == active) return
            voiceModeActive = active
            releaseJob?.cancel()
            releaseJob = null
            warmupJob?.cancel()
            warmupJob = null
            if (!active) {
                scheduleReleaseLocked()
                return
            }
            warmupJob = scope.launch {
                synthesisMutex.withLock {
                    if (isVoiceModeActive()) {
                        // Prewarm is best effort. A later real synthesis can retry initialization.
                        try {
                            getOrCreateEngine(context)
                        } catch (e: Exception) {
                            if (e is kotlinx.coroutines.CancellationException) throw e
                        }
                    }
                }
            }
        }
    }

    private fun isVoiceModeActive(): Boolean = synchronized(lifecycleLock) { voiceModeActive }

    /** Must be called under lifecycleLock. */
    private fun scheduleReleaseLocked() {
        if (voiceModeActive || releaseJob != null) return
        releaseJob = scope.launch {
            try {
                delay(VOICE_EXIT_RELEASE_MS)
                // Wait for any in-progress synthesis before destroying the engine.
                synthesisMutex.withLock {
                    if (!isVoiceModeActive()) {
                        engine?.shutdown()
                        engine = null
                    }
                }
            } finally {
                synchronized(lifecycleLock) {
                    if (releaseJob === coroutineContext[Job]) releaseJob = null
                }
            }
        }
    }

    private fun scheduleReleaseAfterSynthesisIfNeeded() {
        synchronized(lifecycleLock) {
            if (!voiceModeActive) scheduleReleaseLocked()
        }
    }

    /** Must be called under synthesisMutex. */
    private suspend fun getOrCreateEngine(context: Context): TextToSpeech {
        engine?.let { return it }
        val initialized = CompletableDeferred<Int>()
        // Construct on the main thread; the init callback may run on a Binder thread.
        val instance = withContext(Dispatchers.Main.immediate) {
            TextToSpeech(context.applicationContext) { status -> initialized.complete(status) }
        }
        try {
            check(withTimeout(INIT_TIMEOUT_MS) { initialized.await() } == TextToSpeech.SUCCESS) {
                "Failed to initialize Android system TTS engine"
            }
            instance.setOnUtteranceProgressListener(listener)
            engine = instance
            return instance
        } catch (t: Throwable) {
            instance.shutdown()
            throw t
        }
    }

    override fun generateSpeech(
        context: Context,
        providerSetting: TTSProviderSetting.SystemTTS,
        request: TTSRequest
    ): Flow<AudioChunk> = flow {
        val audioData = synthesisMutex.withLock {
            try {
                val tts = getOrCreateEngine(context)
                val locale = Locale.getDefault()
                tts.setLanguage(locale)
                tts.setSpeechRate(providerSetting.speechRate)
                tts.setPitch(providerSetting.pitch)

                val id = UUID.randomUUID().toString()
                val file = File(context.appTempFolder, "tts_$id.wav")
                val task = PendingSynthesis(id)
                pending.set(task)
                try {
                    check(tts.synthesizeToFile(request.text, null, file, id) == TextToSpeech.SUCCESS) {
                        "System TTS could not start synthesis"
                    }
                    withTimeout(SYNTH_TIMEOUT_MS) { task.completed.await() }
                    check(file.isFile && file.length() > 0L) { "System TTS returned an empty WAV" }
                    file.readBytes()
                } catch (t: Throwable) {
                    // A cancelled/failed request must not disrupt a future queued request.
                    runCatching { tts.stop() }
                    throw t
                } finally {
                    pending.compareAndSet(task, null)
                    file.delete()
                }
            } finally {
                scheduleReleaseAfterSynthesisIfNeeded()
            }
        }
        emit(
            AudioChunk(
                data = audioData,
                format = AudioFormat.WAV,
                isLast = true,
                metadata = mapOf(
                    "provider" to "system",
                    "speechRate" to providerSetting.speechRate.toString(),
                    "pitch" to providerSetting.pitch.toString()
                )
            )
        )
    }
}
