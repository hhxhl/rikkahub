package me.rerere.ai.provider.providers.openai

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import me.rerere.ai.provider.RemoteMcpTool
import me.rerere.ai.provider.RemoteMcpApprovalPolicy

internal fun JsonObject.withRemoteMcpTools(servers: List<RemoteMcpTool>): JsonObject {
    if (servers.isEmpty()) return this
    val existing = this["tools"]
    require(existing == null || existing is JsonArray) { "Remote MCP requires a tools array" }
    val remote = servers.map { server ->
        buildJsonObject {
            put("type", "mcp")
            put("server_label", server.label)
            put("server_url", server.url)
            server.authorization?.let { put("authorization", it) }
            if (server.headers.isNotEmpty()) {
                put("headers", buildJsonObject {
                    server.headers.forEach { (name, value) -> put(name, value) }
                })
            }
            when (server.approvalPolicy) {
                RemoteMcpApprovalPolicy.READ_ONLY -> {
                    put("allowed_tools", buildJsonObject { put("read_only", true) })
                    put("require_approval", "never")
                }
                RemoteMcpApprovalPolicy.WRITE_APPROVAL -> put("require_approval", buildJsonObject {
                    // Only explicitly annotated read-only tools bypass approval.
                    put("never", buildJsonObject { put("read_only", true) })
                })
                RemoteMcpApprovalPolicy.NO_APPROVAL -> put("require_approval", "never")
            }
        }
    }
    return JsonObject(this + ("tools" to JsonArray((existing as? JsonArray).orEmpty() + remote)))
}
