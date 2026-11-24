package me.rerere.ai.ui

import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonPrimitive

// Provider-created metadata, not a model-declared function name.
val UIMessagePart.Tool.remoteMcpApprovalRequest: JsonObject?
    get() = (metadata?.get("remote_mcp_approval") as? JsonObject)?.takeIf {
        it["type"]?.jsonPrimitive?.contentOrNull == "mcp_approval_request" &&
            it["id"]?.jsonPrimitive?.contentOrNull == toolCallId
    }

val UIMessagePart.Tool.remoteMcpApprovalBinding: String?
    get() = metadata?.get("remote_mcp_binding")?.jsonPrimitive?.contentOrNull
