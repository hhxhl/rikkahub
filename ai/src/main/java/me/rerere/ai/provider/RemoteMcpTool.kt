package me.rerere.ai.provider

import kotlinx.serialization.Serializable

/** A server-side MCP declaration, not a client function tool. */
@Serializable
data class RemoteMcpTool(
    val label: String,
    val url: String,
    val authorization: String? = null,
    val headers: Map<String, String> = emptyMap(),
) {
    override fun toString(): String = "RemoteMcpTool(label=$label, credentials=REDACTED)"
}
