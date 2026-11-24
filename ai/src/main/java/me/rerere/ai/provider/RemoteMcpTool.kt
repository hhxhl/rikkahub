package me.rerere.ai.provider

import kotlinx.serialization.Serializable

@Serializable
enum class RemoteMcpApprovalPolicy {
    READ_ONLY, WRITE_APPROVAL, NO_APPROVAL,
}

/** A server-side MCP declaration, not a client function tool. */
@Serializable
data class RemoteMcpTool(
    val label: String,
    val url: String,
    val authorization: String? = null,
    val headers: Map<String, String> = emptyMap(),
    val approvalPolicy: RemoteMcpApprovalPolicy = RemoteMcpApprovalPolicy.READ_ONLY,
    val approvalBinding: String = "",
) {
    override fun toString(): String = "RemoteMcpTool(label=$label, credentials=REDACTED)"
}
