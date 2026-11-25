package me.rerere.ai.provider.providers.openai

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import me.rerere.ai.provider.RemoteMcpTool
import me.rerere.ai.provider.RemoteMcpApprovalPolicy
import me.rerere.ai.provider.Model
import me.rerere.ai.provider.TextGenerationParams
import me.rerere.ai.provider.stream.SseEvent
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import me.rerere.ai.ui.ToolApprovalState
import me.rerere.ai.ui.StreamChunk
import me.rerere.ai.ui.StreamChunkHandler
import me.rerere.ai.ui.remoteMcpApprovalRequest
import me.rerere.ai.util.json
import okhttp3.OkHttpClient
import kotlinx.serialization.json.jsonPrimitive
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

    private val approvalRequest = json.parseToJsonElement("""
        {"type":"mcp_approval_request","id":"mcpr_1","server_label":"github","name":"create_branch","arguments":"{\"branch\":\"test\"}"}
    """).jsonObject
    private val server = RemoteMcpTool("github", "https://example.com/mcp",
        approvalPolicy = RemoteMcpApprovalPolicy.WRITE_APPROVAL, approvalBinding = "binding")
    private val api = ResponseAPI(OkHttpClient())

    @Test fun writePoliciesHaveExpectedFilters() {
        val body = buildJsonObject { put("model", "gpt") }
        fun declaration(policy: RemoteMcpApprovalPolicy) =
            body.withRemoteMcpTools(listOf(server.copy(approvalPolicy = policy)))["tools"]!!.jsonArray.single().jsonObject
        val approval = declaration(RemoteMcpApprovalPolicy.WRITE_APPROVAL)
        assertNull(approval["allowed_tools"])
        assertEquals(JsonPrimitive(true), approval["require_approval"]!!.jsonObject["never"]!!.jsonObject["read_only"])
        val automatic = declaration(RemoteMcpApprovalPolicy.NO_APPROVAL)
        assertNull(automatic["allowed_tools"])
        assertEquals(JsonPrimitive("never"), automatic["require_approval"])
    }

    @Test fun approvalDecisionSerializesAsMcpNotFunction() {
        val tool = approvalRequest.toRemoteMcpApprovalTool(mapOf("github" to "binding"))
        assertFalse(tool.isExecuted)
        assertEquals(approvalRequest, tool.remoteMcpApprovalRequest)
        val approved = tool.copy(approvalState = ToolApprovalState.Approved, output = listOf(UIMessagePart.Text("approved")))
        val messages = listOf(UIMessage.user("make a branch"), UIMessage.assistant("").copy(parts = listOf(approved)))
        val input = api.buildMessages(messages)
        assertEquals(approvalRequest, input[input.size - 2])
        val decision = input.last().jsonObject
        assertEquals(JsonPrimitive("mcp_approval_response"), decision["type"])
        assertEquals(JsonPrimitive("mcpr_1"), decision["approval_request_id"])
        assertEquals(JsonPrimitive(true), decision["approve"])
        val denied = approved.copy(approvalState = ToolApprovalState.Denied("stop"))
        assertEquals(JsonPrimitive(false), denied.toRemoteMcpApprovalResponse()["approve"])
        validateRemoteMcpApprovals(messages, listOf(server))
        assertThrows(IllegalArgumentException::class.java) {
            validateRemoteMcpApprovals(messages, listOf(server.copy(approvalBinding = "changed")))
        }
        assertThrows(IllegalArgumentException::class.java) {
            validateRemoteMcpApprovals(messages, emptyList())
        }
        assertThrows(IllegalArgumentException::class.java) {
            validateRemoteMcpApprovals(listOf(messages.first(), messages.last().copy(parts = listOf(approved.copy(input = "{}")))), listOf(server))
        }
    }

    @Test fun streamingApprovalCreatesOneToolAndPreservesArguments() {
        val decoder = ResponseApiStreamDecoder(mapOf("github" to "binding"))
        val item = buildJsonObject {
            put("type", "response.output_item.done")
            put("item", approvalRequest)
        }
        val done = decoder.accept(SseEvent(event = "response.output_item.done", data = item.toString()))
        val terminal = buildJsonObject {
            put("type", "response.completed")
            put("response", buildJsonObject {
                put("id", "resp_1")
                put("status", "completed")
                put("output", JsonArray(listOf(approvalRequest)))
            })
        }
        val final = decoder.accept(SseEvent(event = "response.completed", data = terminal.toString()))
        assertEquals(1, (done.chunks + final.chunks).filterIsInstance<StreamChunk.ToolCallStart>().size)
        val handler = StreamChunkHandler(Model(modelId = "test"))
        val messages = (done.chunks + final.chunks).fold(listOf(UIMessage.user("write"))) { acc, chunk ->
            handler.handle(acc, chunk)
        }
        val tool = messages.last().parts.filterIsInstance<UIMessagePart.Tool>().single()
        assertEquals(approvalRequest, tool.remoteMcpApprovalRequest)
        assertEquals(approvalRequest.getValue("arguments").jsonPrimitive.content, tool.input)
        assertFalse(tool.isExecuted)
    }
}
