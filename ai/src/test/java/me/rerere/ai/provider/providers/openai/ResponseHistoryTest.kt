package me.rerere.ai.provider.providers.openai

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import org.junit.Assert.*
import org.junit.Test

class ResponseHistoryTest {
    private fun item(role: String, text: String) = buildJsonObject {
        put("role", role)
        put("content", text)
    }
    private fun body(vararg items: JsonObject, model: String = "gpt") = buildJsonObject {
        put("model", model)
        put("store", false)
        put("input", JsonArray(items.toList()))
    }

    @Test fun continuationOnlySendsNewInput() {
        val history = ResponseHistory()
        val a = item("user", "a")
        val b = item("assistant", "b")
        val c = item("user", "c")
        val first = history.prepare("session", body(a))
        assertEquals(JsonPrimitive(true), first.body["store"])
        history.remember(first, "resp_1", listOf(b))
        val next = history.prepare("session", body(a, b, c))
        assertTrue(next.continued)
        assertEquals(JsonPrimitive("resp_1"), next.body["previous_response_id"])
        assertEquals(JsonArray(listOf(c)), next.body["input"])
        assertEquals(JsonArray(listOf(a, b, c)), next.fullBody["input"])
        history.invalidate(next)
        assertFalse(history.prepare("session", body(a, b, c)).continued)
    }

    @Test fun editedHistoryAndOtherSessionsCannotContinue() {
        val history = ResponseHistory()
        val a = item("user", "a")
        val b = item("assistant", "b")
        val c = item("user", "c")
        history.remember(history.prepare("session", body(a)), "resp_1", listOf(b))
        assertFalse(history.prepare("other", body(a, b, c)).continued)
        assertFalse(history.prepare("session", body(item("user", "edited"), b, c)).continued)
        assertFalse(history.prepare("session", body(a, b, c, model = "different")).continued)
        assertFalse(history.prepare("session", body(a, b)).continued)
        assertEquals(JsonPrimitive(false), history.prepare(null, body(a)).body["store"])
    }

    @Test fun onlyExplicitMissingResponseErrorsTriggerReplay() {
        assertTrue(isPreviousResponseMissing(400, """{"error":{"code":"previous_response_not_found"}}"""))
        assertTrue(isPreviousResponseMissing(404, """{"error":{"param":"previous_response_id","message":"Response not found"}}"""))
        assertFalse(isPreviousResponseMissing(401, """{"error":{"code":"previous_response_not_found"}}"""))
        assertFalse(isPreviousResponseMissing(404, """{"error":{"message":"Model not found"}}"""))
    }
}
