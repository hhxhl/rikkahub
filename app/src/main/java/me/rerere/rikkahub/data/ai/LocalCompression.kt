package me.rerere.rikkahub.data.ai

import me.rerere.ai.core.MessageRole
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import me.rerere.locallm.litert.LocalContextBudgetException

/**
 * Use the complete textual transcript, including full saved tool outputs, instead of
 * LiteRT's normal trimmed/tool-compacted generation view. Media are identified as
 * attachments; the original conversation nodes are not removed by compression.
 */
@Suppress("DEPRECATION")
internal fun localCompressionTranscript(messages: List<UIMessage>): String {
    fun partText(part: UIMessagePart): String = when (part) {
        is UIMessagePart.Text -> part.text
        is UIMessagePart.Notice -> part.text
        is UIMessagePart.Tool -> "Tool ${part.toolName} arguments: ${part.input}\nResult:\n" +
            part.output.joinToString("\n") { partText(it) }
        is UIMessagePart.ToolCall -> "Tool ${part.toolName} arguments: ${part.arguments}"
        is UIMessagePart.ToolResult -> "Tool ${part.toolName} result: ${part.content}"
        is UIMessagePart.ServerTool -> "Tool ${part.toolName}: ${part.input}\n${part.output}"
        is UIMessagePart.Image -> "[Image attachment]"
        is UIMessagePart.Audio -> "[Audio attachment]"
        is UIMessagePart.Video -> "[Video attachment]"
        is UIMessagePart.Document -> "[Document: ${part.fileName}]"
        else -> "" // Internal reasoning and legacy search marker are not conversation facts.
    }
    return messages.filterNot { it.role == MessageRole.SYSTEM && it.isSynthetic }
        .joinToString("\n\n") { message ->
            "${message.role}:\n" + message.parts.joinToString("\n") { partText(it) }
        }
}

/**
 * Submit one user message at a time, so the Provider cannot silently drop older
 * turns. Split only on a known budget failure; carry each successful summary
 * forward. If even a minimal chunk cannot fit, fail without creating a checkpoint.
 */
internal suspend fun summarizeLocalTranscript(
    transcript: String,
    request: suspend (chunk: String, previous: String?) -> String,
): String {
    suspend fun summarize(chunk: String, previous: String?): String {
        try {
            return request(chunk, previous)
        } catch (e: LocalContextBudgetException) {
            if (chunk.codePointCount(0, chunk.length) <= 1) {
                throw IllegalStateException("Local context is too small for the compression instructions and accumulated summary", e)
            }
            var split = chunk.length / 2
            if (Character.isLowSurrogate(chunk[split])) split--
            val newline = chunk.lastIndexOf('\n', split)
            if (newline >= chunk.length / 4 && newline > 0) split = newline + 1
            if (split <= 0 || split >= chunk.length) split = chunk.offsetByCodePoints(0, 1)
            val left = summarize(chunk.substring(0, split), previous)
            return summarize(chunk.substring(split), left)
        }
    }
    return summarize(transcript, null)
}
