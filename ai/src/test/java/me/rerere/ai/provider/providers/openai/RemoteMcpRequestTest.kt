package me.rerere.ai.provider.providers.openai

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import me.rerere.ai.provider.RemoteMcpTool
import org.junit.Assert.*
import org.junit.Test

class RemoteMcpRequestTest {
    @Test fun appendsReadOnlyMcpWithoutReplacingExistingTools() {
        val existing = buildJsonObject { put("type", "web_search") }
        val body = buildJsonObject { put("tools", JsonArray(listOf(existing))) }
        val server = RemoteMcpTool("github", "https://example.com/mcp", "secret", mapOf("X-MCP-Toolsets" to "repos"))
        val result = body.withRemoteMcpTools(listOf(server))["tools"]!!.jsonArray
        assertEquals(existing, result.first())
        val tool = result.last().jsonObject
        assertEquals(JsonPrimitive("mcp"), tool["type"])
        assertEquals(JsonPrimitive("github"), tool["server_label"])
        assertEquals(JsonPrimitive(true), tool["allowed_tools"]!!.jsonObject["read_only"])
        assertEquals(JsonPrimitive("never"), tool["require_approval"])
        assertEquals(JsonPrimitive("secret"), tool["authorization"])
        assertFalse(server.toString().contains("secret"))
    }

    @Test fun emptyListLeavesRequestUnchanged() {
        val body = buildJsonObject { put("model", "gpt") }
        assertEquals(body, body.withRemoteMcpTools(emptyList()))
    }

    @Test fun discoveryIsServerTool() {
        assertTrue(isOpenAIServerToolCall("mcp_list_tools"))
        assertTrue(isOpenAIServerToolCall("mcp_call"))
        assertFalse(isOpenAIServerToolCall("function_call"))
    }
}
