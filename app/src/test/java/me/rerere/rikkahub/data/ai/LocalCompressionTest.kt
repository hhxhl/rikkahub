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
    fun `summary limit follows actual local capacity`() {
        assertEquals(160, localSummaryTokenLimit(1280))
        assertEquals(512, localSummaryTokenLimit(4096))
        assertEquals(1024, localSummaryTokenLimit(32768))
    }

    @Test
    fun `step count survives compression checkpoint and ignores previous user turn`() {
        val old = UIMessage(role = MessageRole.ASSISTANT, parts = emptyList(), stepCount = 9)
        val first = UIMessage(role = MessageRole.ASSISTANT, parts = emptyList(), stepCount = 3)
        val second = UIMessage(role = MessageRole.ASSISTANT, parts = emptyList(), stepCount = 2)
        val history = listOf(UIMessage.user("old"), old, UIMessage.user("current"),
            first, UIMessage.user("summary").copy(isContextCheckpoint = true), second)
        assertEquals(5, completedStepsInUserTurn(history, true))
        assertEquals(0, completedStepsInUserTurn(history + UIMessage.user("new"), false))
    }

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
