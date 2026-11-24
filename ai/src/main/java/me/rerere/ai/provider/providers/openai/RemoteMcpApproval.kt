package me.rerere.ai.provider.providers.openai

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import me.rerere.ai.provider.RemoteMcpTool
import me.rerere.ai.ui.UIMessage
import me.rerere.ai.ui.UIMessagePart
import me.rerere.ai.ui.ToolApprovalState
import me.rerere.ai.ui.remoteMcpApprovalRequest
import me.rerere.ai.ui.remoteMcpApprovalBinding
import me.rerere.ai.ui.ServerToolMetadata
import me.rerere.ai.ui.metadataAs

internal fun JsonObject.toRemoteMcpApprovalTool(bindings: Map<String, String>): UIMessagePart.Tool {
    val id = getValue("id").jsonPrimitive.content
    val label = getValue("server_label").jsonPrimitive.content
    val name = getValue("name").jsonPrimitive.content
    val binding = bindings[label]?.takeIf { it.isNotBlank() }
        ?: error("Remote MCP approval is not associated with a configured server")
    val arguments = getValue("arguments").jsonPrimitive.content
    val raw = JsonObject(filterKeys { it in setOf("type", "id", "server_label", "name", "arguments") })
    return UIMessagePart.Tool(
        toolCallId = id,
        toolName = "MCP: $label / $name",
        input = arguments,
        metadata = buildJsonObject {
            put("remote_mcp_approval", raw)
            put("remote_mcp_binding", binding)
        },
    )
}

internal fun UIMessagePart.Tool.toRemoteMcpApprovalResponse(): JsonObject {
    require(remoteMcpApprovalRequest != null)
    require(isExecuted)
    val approved = when (approvalState) {
        ToolApprovalState.Approved -> true
        is ToolApprovalState.Denied -> false
        else -> error("Remote MCP approval is not decided")
    }
    return buildJsonObject {
        put("type", "mcp_approval_response")
        put("approval_request_id", toolCallId)
        put("approve", approved)
        (approvalState as? ToolApprovalState.Denied)?.reason?.takeIf { it.isNotBlank() }?.let { put("reason", it) }
    }
}

internal fun validateRemoteMcpApprovals(messages: List<UIMessage>, servers: List<RemoteMcpTool>) {
    val last = messages.lastOrNull() ?: return
    val completed = last.parts.filterIsInstance<UIMessagePart.ServerTool>()
        .mapNotNull { it.metadataAs<ServerToolMetadata>()?.call?.get("approval_request_id")?.jsonPrimitive?.contentOrNull }.toSet()
    // Only outstanding approvals may cause new execution; historical resolved
    // requests remain ordinary context. Do not authorize on another server.
    for (tool in last.parts.filterIsInstance<UIMessagePart.Tool>()) {
        val raw = tool.remoteMcpApprovalRequest ?: continue
        if (tool.toolCallId in completed || !tool.isExecuted) continue
        require(raw.getValue("arguments").jsonPrimitive.content == tool.input) {
            "Remote MCP approval arguments changed. Request approval again."
        }
        val label = raw.getValue("server_label").jsonPrimitive.content
        val expected = servers.firstOrNull { it.label == label }?.approvalBinding
        require(!expected.isNullOrBlank() && expected == tool.remoteMcpApprovalBinding) {
            "Remote MCP configuration changed. Start a new tool request instead of reusing this approval."
        }
    }
}
