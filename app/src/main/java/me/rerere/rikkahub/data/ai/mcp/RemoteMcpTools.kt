package me.rerere.rikkahub.data.ai.mcp

import me.rerere.ai.provider.ProviderSetting
import me.rerere.ai.provider.RemoteMcpTool
import okhttp3.HttpUrl.Companion.toHttpUrlOrNull
import kotlin.uuid.Uuid

internal fun remoteMcpTools(
    servers: List<McpServerConfig>,
    selected: Set<Uuid>,
    provider: ProviderSetting,
): List<RemoteMcpTool> {
    val enabled = servers.filter {
        it.id in selected && it.commonOptions.enable &&
            it.commonOptions.executionMode == McpExecutionMode.SERVER
    }
    if (enabled.isEmpty()) return emptyList()
    // Only the official OpenAI API has been targeted in this first version.
    require(provider is ProviderSetting.OpenAI && provider.useResponseApi &&
        provider.baseUrl.toHttpUrlOrNull()?.let { it.isHttps && it.host == "api.openai.com" } == true) {
        "Server MCP requires the official OpenAI Responses API. Use client execution for other providers."
    }
    val names = enabled.map { it.commonOptions.name }
    require(names.all { it.matches(Regex("[A-Za-z0-9_-]{1,64}")) } && names.distinct().size == names.size) {
        "Server MCP requires unique labels (letters, digits, underscores or hyphens)."
    }
    return enabled.map { server ->
        val url = server.serverUrl.toHttpUrlOrNull()
        require(url != null && url.isHttps && url.username.isEmpty() && url.password.isEmpty()) {
            "Server MCP requires an HTTPS server URL without embedded credentials."
        }
        require(server.commonOptions.oauth?.enabled != true) {
            "Server MCP OAuth is not supported yet. Configure a read-only token in request headers."
        }
        val headers = linkedMapOf<String, String>()
        var authorization: String? = null
        for ((name, value) in server.commonOptions.headers) {
            if (name.isBlank()) continue
            if (name.equals("Authorization", ignoreCase = true) && value.startsWith("Bearer ", ignoreCase = true)) {
                authorization = value.substringAfter(' ').trim().takeIf { it.isNotEmpty() }
            } else {
                headers[name] = value
            }
        }
        RemoteMcpTool(server.commonOptions.name, url.toString(), authorization, headers)
    }
}
