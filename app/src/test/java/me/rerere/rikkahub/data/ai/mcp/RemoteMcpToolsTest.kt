package me.rerere.rikkahub.data.ai.mcp

import me.rerere.ai.provider.ProviderSetting
import me.rerere.ai.provider.RemoteMcpApprovalPolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class RemoteMcpToolsTest {
    private val server = McpServerConfig.StreamableHTTPServer(
        url = "https://example.com/mcp",
        commonOptions = McpCommonOptions(
            name = "github",
            executionMode = McpExecutionMode.SERVER,
            remoteApprovalPolicy = RemoteMcpApprovalPolicy.WRITE_APPROVAL,
            headers = listOf("Authorization" to "Bearer test-token"),
        ),
    )

    private val provider = ProviderSetting.OpenAI(
        baseUrl = "https://gateway.example.com/v1",
        apiKey = "test-key",
        useResponseApi = true,
    )

    private fun tools(provider: ProviderSetting) =
        remoteMcpTools(listOf(server), setOf(server.id), provider, "test-model")

    @Test fun compatibleThirdPartyEndpointIsAllowed() {
        val tool = tools(provider).single()
        assertEquals("test-token", tool.authorization)
        assertEquals(RemoteMcpApprovalPolicy.WRITE_APPROVAL, tool.approvalPolicy)
        assertNotEquals(tool.approvalBinding, tools(provider.copy(responsesPath = "/custom/responses")).single().approvalBinding)
    }

    @Test fun chatCompletionsAndInsecureEndpointsAreRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            tools(provider.copy(useResponseApi = false))
        }
        assertThrows(IllegalArgumentException::class.java) {
            tools(provider.copy(baseUrl = "http://gateway.example.com/v1"))
        }
    }

    @Test fun defaultExecutionModeIsClient() {
        assertEquals(McpExecutionMode.CLIENT, McpCommonOptions().executionMode)
    }
}
