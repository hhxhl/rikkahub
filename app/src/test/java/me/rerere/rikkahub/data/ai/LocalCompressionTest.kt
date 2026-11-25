package me.rerere.rikkahub.data.ai

import kotlinx.coroutines.test.runTest
import me.rerere.ai.core.MessageRole
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import me.rerere.locallm.litert.LocalContextBudgetException
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LocalCompressionTest {
    @Test
    fun `rolling summary visits every chunk in order`() = runTest {
        val input = "一二三四😀五六七八"
        val visited = mutableListOf<String>()
        val result = summarizeLocalTranscript(input) { chunk, previous ->
            if (chunk.length > 4) throw LocalContextBudgetException("too long")
            visited += chunk
            previous.orEmpty() + chunk
        }
        assertEquals(input, visited.joinToString(""))
        assertEquals(input, result)
    }

    @Test
    fun `full stored tool results are included in compression`() {
        val output = "正文".repeat(2000)
        val message = UIMessage(role = MessageRole.ASSISTANT, parts = listOf(
            UIMessagePart.Tool(toolCallId = "call", toolName = "scrape_web", input = "{}", output = listOf(UIMessagePart.Text(output))),
        ))
        assertTrue(localCompressionTranscript(listOf(message)).contains(output))
    }
}
