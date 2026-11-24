package me.rerere.ai.provider.providers.openai

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import me.rerere.ai.provider.ProviderSetting
import me.rerere.ai.provider.TextGenerationParams
import me.rerere.ai.util.json
import java.security.MessageDigest

/**
 * Bounded, process-local checkpoints. Only response IDs and hashes of wire input
 * are kept. An edited/truncated history or changed request configuration cannot
 * accidentally attach to an unrelated server-side conversation.
 */
internal class ResponseHistory(private val capacity: Int = 32) {
    private data class Checkpoint(val id: String, val signature: String, val prefix: List<String>)
    private val checkpoints = LinkedHashMap<String, Checkpoint>(16, 0.75f, true)

    data class Attempt(
        val scope: String?,
        val fullBody: JsonObject,
        val body: JsonObject,
        val previousId: String? = null,
    ) {
        val continued: Boolean get() = previousId != null
    }

    @Synchronized
    fun prepare(scope: String?, body: JsonObject): Attempt {
        if (scope == null || body["input"] !is JsonArray) return Attempt(null, body, body)
        val full = JsonObject(body + ("store" to JsonPrimitive(true)))
        val input = full["input"] as JsonArray
        val previous = checkpoints[scope]
        val hashes = input.map { fingerprint(it.toString()) }
        if (previous != null && previous.signature == signature(full) &&
            hashes.size > previous.prefix.size &&
            hashes.take(previous.prefix.size) == previous.prefix
        ) {
            val delta = JsonObject(full + mapOf(
                "previous_response_id" to JsonPrimitive(previous.id),
                "input" to JsonArray(input.drop(previous.prefix.size)),
            ))
            return Attempt(scope, full, delta, previous.id)
        }
        return Attempt(scope, full, full)
    }

    @Synchronized
    fun remember(attempt: Attempt, responseId: String, output: List<JsonElement>) {
        val scope = attempt.scope ?: return
        if (responseId.isBlank() || output.isEmpty()) return
        val input = attempt.fullBody["input"] as? JsonArray ?: return
        checkpoints[scope] = Checkpoint(
            responseId, signature(attempt.fullBody),
            (input + output).map { fingerprint(it.toString()) },
        )
        while (checkpoints.size > capacity.coerceAtLeast(1)) {
            checkpoints.remove(checkpoints.keys.first())
        }
    }

    @Synchronized
    fun invalidate(attempt: Attempt) {
        val scope = attempt.scope ?: return
        if (checkpoints[scope]?.id == attempt.previousId) checkpoints.remove(scope)
    }

    private fun signature(body: JsonObject): String = fingerprint(
        JsonObject(body.filterKeys { it !in setOf("input", "stream", "store", "previous_response_id", "instructions") }).toString()
    )
}

internal fun responseHistoryScope(provider: ProviderSetting.OpenAI, params: TextGenerationParams): String? {
    if (!provider.useResponseApi || !provider.serverManagedHistory) return null
    val session = params.sessionId?.takeIf { it.isNotBlank() } ?: return null
    // A rotated key may belong to a different account/project.
    val key = provider.apiKey.trim()
    if (key.isBlank() || key.contains(Regex("[\\s,]"))) return null
    // Explicit custom state/input overrides belong to the caller.
    if (params.customBody.any { it.key in setOf("input", "store", "previous_response_id", "conversation", "model") }) return null
    return fingerprint(listOf(provider.id.toString(), provider.baseUrl, provider.responsesPath,
        key, session, params.model.id.toString(), params.customHeaders.toString()).joinToString("\u0000"))
}

private fun fingerprint(value: String): String =
    MessageDigest.getInstance("SHA-256").digest(value.toByteArray(Charsets.UTF_8))
        .joinToString("") { "%02x".format(it.toInt() and 0xff) }

internal class PreviousResponseMissing : IllegalStateException("Previous response is unavailable")

internal fun isPreviousResponseMissing(status: Int?, raw: String?): Boolean {
    if (status !in setOf(400, 404) || raw.isNullOrBlank()) return false
    return runCatching {
        val error = json.parseToJsonElement(raw).jsonObject["error"]?.jsonObject ?: return false
        val code = error["code"]?.jsonPrimitive?.contentOrNull
        val param = error["param"]?.jsonPrimitive?.contentOrNull
        val message = error["message"]?.jsonPrimitive?.contentOrNull.orEmpty().lowercase()
        code == "previous_response_not_found" ||
            (param == "previous_response_id" && ("not found" in message || "expired" in message || "does not exist" in message)) ||
            ("previous response" in message && ("not found" in message || "expired" in message))
    }.getOrDefault(false)
}
