package me.rerere.rikkahub.data.ai

import me.rerere.common.android.LogEntry
import me.rerere.common.android.Logging
import okhttp3.HttpUrl
import okhttp3.Interceptor
import okhttp3.Response
import okio.Buffer

class RequestLoggingInterceptor : Interceptor {
    override fun intercept(chain: Interceptor.Chain): Response {
        if (!Logging.isRequestLoggingEnabled()) {
            return chain.proceed(chain.request())
        }

        val request = chain.request()
        val startTime = System.currentTimeMillis()

        val requestHeaders = request.headers.toMap()
        val requestBody = request.body?.let { body ->
            val buffer = Buffer()
            body.writeTo(buffer)
            buffer.readUtf8()
        }

        val response: Response
        var error: String? = null

        try {
            response = chain.proceed(request)
        } catch (e: Exception) {
            error = e.message
            Logging.logRequest(
                LogEntry.RequestLog(
                    tag = "HTTP",
                    url = request.url.toRedactedLogUrl(),
                    method = request.method,
                    requestHeaders = requestHeaders,
                    requestBody = requestBody,
                    error = error
                )
            )
            throw e
        }

        val durationMs = System.currentTimeMillis() - startTime
        val responseHeaders = response.headers.toMap()

        Logging.logRequest(
            LogEntry.RequestLog(
                tag = "HTTP",
                url = request.url.toRedactedLogUrl(),
                method = request.method,
                requestHeaders = requestHeaders,
                requestBody = requestBody,
                responseCode = response.code,
                responseHeaders = responseHeaders,
                durationMs = durationMs,
                error = error
            )
        )

        return response
    }

    private fun okhttp3.Headers.toMap(): Map<String, String> {
        return names().associateWith { name ->
            if (name.lowercase() in SENSITIVE_HEADERS) {
                "██"
            } else {
                get(name) ?: ""
            }
        }
    }

    private fun HttpUrl.toRedactedLogUrl(): String {
        val builder = newBuilder()
        if (username.isNotEmpty()) builder.username("REDACTED")
        if (password.isNotEmpty()) builder.password("REDACTED")
        queryParameterNames
            .filter { it.lowercase() in SENSITIVE_QUERY_PARAMETERS }
            .forEach { builder.setQueryParameter(it, "REDACTED") }
        return builder.build().toString()
    }

    private companion object {
        val SENSITIVE_HEADERS = setOf(
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "x-goog-api-key",
            "api-key",
            "x-auth-token",
            "x-access-token",
            "cookie",
            "set-cookie",
            "x-amz-security-token",
        )

        val SENSITIVE_QUERY_PARAMETERS = setOf(
            "key",
            "api_key",
            "apikey",
            "access_token",
            "auth_token",
            "token",
            "authorization",
            "client_secret",
            "signature",
        )
    }
}
