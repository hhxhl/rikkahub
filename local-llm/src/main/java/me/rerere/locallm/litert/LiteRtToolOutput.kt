package me.rerere.locallm.litert

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put

/**
 * Only the text sent to LiteRT is compacted. UIMessagePart.Tool.output (and thus cloud
 * providers, saved transcript, search cards and citation IDs) keeps the original JSON.
 *
 * Never feed the lengthy shared tool description or its example citation URLs back to a
 * small local model. Likewise, avoid presenting the entire search API response as a
 * sample answer: a small model may reproduce it instead of answering the user.
 */
internal fun compactLocalToolOutput(toolName: String, rawOutput: String): String = when (toolName) {
    "search_web" -> compactLocalSearchOutput(rawOutput)
    else -> rawOutput
}

internal fun compactLocalSearchOutput(rawOutput: String): String {
    val payload = runCatching { Json.parseToJsonElement(rawOutput).jsonObject }.getOrNull()
        ?: return rawOutput.take(500)
    val items = (payload["items"] as? JsonArray).orEmptyItems()
    return buildJsonObject {
        // Provider-supplied answer can be useful, but not if it dominates 4096-token context.
        payload["answer"].stringOrNull()?.takeIf(String::isNotBlank)?.let {
            put("answer", it.take(320))
        }
        put("items", JsonArray(items.take(3).mapNotNull { element ->
            val item = element as? JsonObject ?: return@mapNotNull null
            buildJsonObject {
                for (key in listOf("id", "title", "url", "publishedDate")) {
                    item[key].stringOrNull()?.let { put(key, it) }
                }
                item["text"].stringOrNull()?.takeIf(String::isNotBlank)?.let {
                    put("text", it.take(280))
                }
                val highlights = (item["highlights"] as? JsonArray).orEmptyItems()
                    .mapNotNull { it.stringOrNull() }.take(2)
                    .map { JsonPrimitive(it.take(120)) }
                if (highlights.isNotEmpty()) put("highlights", JsonArray(highlights))
            }
        }))
    }.toString()
}

private fun JsonElement?.stringOrNull(): String? = (this as? JsonPrimitive)
    ?.takeIf { it.isString }
    ?.content

private fun JsonArray?.orEmptyItems(): List<JsonElement> = this?.toList().orEmpty()
