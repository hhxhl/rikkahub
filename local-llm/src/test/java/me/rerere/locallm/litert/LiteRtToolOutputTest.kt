package me.rerere.locallm.litert

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtToolOutputTest {
    @Test
    fun `search result retains real citation and URL without bulky metadata`() {
        val original = """{"retrievedAt":"2026-11-15T23:00:00Z","answer":"Answer","items":[{"id":"abc123","index":1,"title":"Android Developers","url":"https://developer.android.com/","publishedDate":"2026-11-15","text":"${"x".repeat(600)}","highlights":["${"h".repeat(200)}","useful"],"extra":"discard"}],"images":["https://example.invalid/unused.png"]}"""
        val compact = compactLocalToolOutput("search_web", original)
        val parsed = Json.parseToJsonElement(compact).jsonObject
        assertFalse("retrievedAt" in parsed)
        assertFalse("images" in parsed)
        val item = parsed["items"]!!.jsonArray.single().jsonObject
        assertEquals("abc123", item["id"]!!.jsonPrimitive.content)
        assertEquals("https://developer.android.com/", item["url"]!!.jsonPrimitive.content)
        assertEquals(280, item["text"]!!.jsonPrimitive.content.length)
        assertFalse("extra" in item)
        assertTrue(compact.length < original.length)
    }

    @Test
    fun `compact search caps local results but leaves other tool outputs alone`() {
        val original = """{"items":[${(1..7).joinToString(",") { """{"id":"id$it","url":"https://source$it.invalid","title":"$it","text":"body"}""" }}]}"""
        val compact = compactLocalToolOutput("search_web", original)
        assertEquals(3, Json.parseToJsonElement(compact).jsonObject["items"]!!.jsonArray.size)
        assertEquals(original, compactLocalToolOutput("get_time_info", original))
    }

    @Test
    fun `malformed search output is safely bounded`() {
        val raw = "invalid".repeat(900)
        assertEquals(500, compactLocalToolOutput("search_web", raw).length)
    }
}
