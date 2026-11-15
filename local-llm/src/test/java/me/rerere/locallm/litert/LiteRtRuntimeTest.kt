package me.rerere.locallm.litert

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-decision-function tests for [LiteRtRuntime] — the two companion-object helpers
 * are load-bearing on every model load but were untested in 22A. Closes that gap.
 *
 * `planTurns`: governs whether the warm Conversation's KV cache can be reused for the
 * next turn (Warm) or whether we must rebuild it cold. A wrong answer can never produce
 * incorrect output (the cold path is always correct), but it can either cost a missed
 * optimisation OR cost a correctness bug if the warm path is taken when it shouldn't.
 *
 * `isVisionExecutorError`: governs the vision-encoder text-only fallback path. A false
 * negative means the user sees a hard "engine could not load" instead of a working
 * text-only Gemma 4; a false positive would mis-route an unrelated load failure into
 * the text-only retry and waste an init attempt. Both cost user-visible failures.
 */
class LiteRtRuntimeTest {

    // ---- planTurns -----------------------------------------------------------------

    private val u1 = turnSignature(ROLE_USER, "hello")
    private val a1 = turnSignature(ROLE_ASSISTANT, "hi there")
    private val u2 = turnSignature(ROLE_USER, "what's up")
    private val a2 = turnSignature(ROLE_ASSISTANT, "not much")
    private val u3 = turnSignature(ROLE_USER, "tell me a joke")

    @Test
    fun `planTurns returns Cold when processed is empty (first turn ever)`() {
        val plan = LiteRtRuntime.planTurns(
            processed = emptyList(),
            historySignatures = listOf(u1),
            hasMedia = false,
        )
        assertEquals(TurnPlan.Cold, plan)
    }

    @Test
    fun `planTurns returns Cold when caller has media inputs`() {
        // Media kills the warm path because there's no way to attach per-call images onto
        // a prior turn in the KV cache.
        val plan = LiteRtRuntime.planTurns(
            processed = listOf(u1, a1),
            historySignatures = listOf(u1, a1, u2),
            hasMedia = true,
        )
        assertEquals(TurnPlan.Cold, plan)
    }

    @Test
    fun `planTurns returns Warm for a clean single-turn append on top of prior history`() {
        // The prior turn finished, the runtime recorded [u1, assistant_reply]; the caller
        // now wants to send [u1, assistant_reply, u2] — exactly one new turn appended.
        val plan = LiteRtRuntime.planTurns(
            processed = listOf(u1, a1),
            historySignatures = listOf(u1, a1, u2),
            hasMedia = false,
        )
        assertTrue("expected Warm got $plan", plan is TurnPlan.Warm)
        assertEquals(2, (plan as TurnPlan.Warm).sendFromIndex)
    }

    @Test
    fun `planTurns returns Cold when more than one turn was appended`() {
        // Two new turns at once (e.g. resumed conversation, batched input) — can't reuse.
        val plan = LiteRtRuntime.planTurns(
            processed = listOf(u1, a1),
            historySignatures = listOf(u1, a1, u2, a2, u3),
            hasMedia = false,
        )
        assertEquals(TurnPlan.Cold, plan)
    }

    @Test
    fun `planTurns returns Cold when an earlier turn was rewritten`() {
        // User edited turn u1 to u1'. The prefix no longer matches.
        val u1Edited = turnSignature(ROLE_USER, "hello there (edited)")
        val plan = LiteRtRuntime.planTurns(
            processed = listOf(u1, a1),
            historySignatures = listOf(u1Edited, a1, u2),
            hasMedia = false,
        )
        assertEquals(TurnPlan.Cold, plan)
    }

    @Test
    fun `planTurns returns Cold when caller history is shorter than processed (regenerate)`() {
        // User hit "regenerate" — history rolled back. Same prefix but fewer turns than
        // the runtime has consumed. Force cold so the new turn is generated from scratch.
        val plan = LiteRtRuntime.planTurns(
            processed = listOf(u1, a1, u2, a2),
            historySignatures = listOf(u1, a1, u2),
            hasMedia = false,
        )
        assertEquals(TurnPlan.Cold, plan)
    }

    @Test
    fun `planTurns returns Cold when caller history exactly equals processed (no new turn)`() {
        // Defensive: a caller that sends the same history without appending should not
        // trigger a "warm" path with sendFromIndex == processed.size, which would IndexError.
        val plan = LiteRtRuntime.planTurns(
            processed = listOf(u1, a1),
            historySignatures = listOf(u1, a1),
            hasMedia = false,
        )
        assertEquals(TurnPlan.Cold, plan)
    }

    @Test
    fun `turnSignature is stable for identical role and content`() {
        assertEquals(
            turnSignature(ROLE_USER, "hello"),
            turnSignature(ROLE_USER, "hello"),
        )
    }

    @Test
    fun `turnSignature differs when role differs`() {
        assertFalse(turnSignature(ROLE_USER, "hi") == turnSignature(ROLE_ASSISTANT, "hi"))
    }

    @Test
    fun `turnSignature differs when content differs`() {
        assertFalse(turnSignature(ROLE_USER, "a") == turnSignature(ROLE_USER, "b"))
    }

    @Test
    fun `first Cold turn can keep a pristine Conversation`() {
        assertFalse(LiteRtRuntime.shouldRecreateForCold(pristineConversation = true))
    }

    @Test
    fun `cancelled or previously used Conversation must be rebuilt on Cold`() {
        assertTrue(LiteRtRuntime.shouldRecreateForCold(pristineConversation = false))
    }

    @Test
    fun `a model switch must wait when ensureLoaded has not started streaming`() {
        assertTrue(LiteRtRuntime.shouldDeferModelSwitch(pendingStream = true))
        assertFalse(LiteRtRuntime.shouldDeferModelSwitch(pendingStream = false))
    }

    @Test
    fun `native Cold recovery preserves roles and excludes newest USER`() {
        val history = listOf(
            Turn(ROLE_USER, "hello"),
            Turn(ROLE_ASSISTANT, "hi"),
            Turn(ROLE_USER, "current question"),
        )
        val native = LiteRtRuntime.nativeInitialHistory(history)!!
        assertEquals(2, native.size)
        assertEquals(com.google.ai.edge.litertlm.Role.USER, native[0].role)
        assertEquals(com.google.ai.edge.litertlm.Role.MODEL, native[1].role)
        assertEquals("hello", native[0].toString())
        assertEquals("hi", native[1].toString())
    }

    @Test
    fun `tool continuation does not masquerade as a new USER`() {
        assertEquals(
            null,
            LiteRtRuntime.nativeInitialHistory(
                listOf(Turn(ROLE_USER, "find news"), Turn(ROLE_ASSISTANT, "<tool_result>ok</tool_result>")),
            ),
        )
    }

    // ---- Native TOOL response continuation ---------------------------------------

    private val searchCall = RuntimeToolCall(
        id = "sdk-tool-search",
        name = "search_web",
        argumentsJson = """{"query":"Android 17"}""",
    )
    private val timeCall = RuntimeToolCall(
        id = "sdk-tool-time",
        name = "get_time_info",
        argumentsJson = "{}",
    )

    private val toolAnchor = listOf(Turn(ROLE_USER, "search and check the time"))
    private val toolStep = toolAnchor + Turn(ROLE_ASSISTANT, "<tool_call>...<tool_result>...")

    @Test
    fun `native results require exact pending ids but allow earlier completed tool steps`() {
        val batch = PendingToolBatch(toolAnchor.map { it.signature }, listOf(searchCall, timeCall))
        val earlier = RuntimeToolResult("previous-id", "search_web", "{}", "previous output")
        val time = RuntimeToolResult(timeCall.id, timeCall.name, "{}", "2026-11-16 05:02")
        val search = RuntimeToolResult(searchCall.id, searchCall.name, """{ "query" : "Android 17" }""", "official URL")
        assertEquals(listOf(search, time), matchNativeToolResults(batch, toolStep, listOf(earlier, time, search)))
    }

    @Test
    fun `native tool response uses SDK TOOL role and preserves output text`() {
        val results = listOf(RuntimeToolResult(searchCall.id, searchCall.name, searchCall.argumentsJson, "actual result"))
        val message = nativeToolResponseMessage(results)
        assertEquals(com.google.ai.edge.litertlm.Role.TOOL, message.role)
        val content = message.contents.contents.single() as com.google.ai.edge.litertlm.Content.ToolResponse
        assertEquals("search_web", content.name)
        assertEquals("actual result", content.response)
    }

    @Test
    fun `native results reject missing duplicate mismatched calls and changed history`() {
        val batch = PendingToolBatch(toolAnchor.map { it.signature }, listOf(searchCall, timeCall))
        val search = RuntimeToolResult(searchCall.id, searchCall.name, searchCall.argumentsJson, "search result")
        val time = RuntimeToolResult(timeCall.id, timeCall.name, "{}", "time result")
        assertEquals(null, matchNativeToolResults(batch, toolStep, listOf(search)))
        assertEquals(null, matchNativeToolResults(batch, toolStep, listOf(search, search, time)))
        assertEquals(null, matchNativeToolResults(batch, toolStep, listOf(search.copy(name = "scrape_web"), time)))
        assertEquals(null, matchNativeToolResults(batch, toolStep, listOf(search.copy(argumentsJson = "{}"), time)))
        assertEquals(null, matchNativeToolResults(batch, toolStep, listOf(search.copy(output = ""), time)))
        assertEquals(null, matchNativeToolResults(batch, listOf(Turn(ROLE_USER, "edited"), toolStep.last()), listOf(search, time)))
        assertEquals(null, matchNativeToolResults(batch, toolAnchor, listOf(search, time)))
        assertEquals(null, matchNativeToolResults(null, toolStep, listOf(search, time)))
    }

    // ---- isVisionExecutorError -----------------------------------------------------

    @Test
    fun `isVisionExecutorError matches the canonical SDK file path string`() {
        // The full native error text the SDK throws on the broken-vision device.
        val msg = "Failed to create engine: INTERNAL: ERROR: " +
            "[third_party/odml/litert_lm/runtime/executor/vision_litert_compiled_model_executor.cc:273]"
        assertTrue(LiteRtRuntime.isVisionExecutorError(msg))
    }

    @Test
    fun `isVisionExecutorError matches the truncated form`() {
        // Some SDK builds truncate the file path before `_compiled_model`. Match the
        // shorter `vision_litert` prefix too.
        val msg = "vision_litert error during init"
        assertTrue(LiteRtRuntime.isVisionExecutorError(msg))
    }

    @Test
    fun `isVisionExecutorError matches CreateSharedMemoryManager UNIMPLEMENTED (upstream root cause)`() {
        // The underlying OpenGL fallback stub from upstream LiteRT-LM #2292.
        val msg = "UNIMPLEMENTED: CreateSharedMemoryManager is not implemented."
        assertTrue(LiteRtRuntime.isVisionExecutorError(msg))
    }

    @Test
    fun `isVisionExecutorError matches gpu_backend_opengl source path`() {
        val msg = "third_party/odml/litert/ml_drift/delegate/gpu_backend_opengl.cc:169"
        assertTrue(LiteRtRuntime.isVisionExecutorError(msg))
    }

    @Test
    fun `isVisionExecutorError is case-insensitive`() {
        assertTrue(LiteRtRuntime.isVisionExecutorError("Vision_LiteRT_Compiled_Model_Executor blew up"))
        assertTrue(LiteRtRuntime.isVisionExecutorError("createSHAREDmemorymanager"))
    }

    @Test
    fun `isVisionExecutorError returns false for unrelated load failures`() {
        // Common false-positive candidates.
        val unrelated = listOf(
            "Invalid magic number while reading model file",
            "No KV cache inputs found",
            "FAILED_PRECONDITION: tokenizer section missing",
            "Input token ids are too long. Exceeding the maximum: 5000 >= 4096",
            "Status Code: 3. Message: tokenizer mismatch",
            "OutOfMemoryError",
        )
        for (msg in unrelated) {
            assertFalse(
                "isVisionExecutorError should not match: $msg",
                LiteRtRuntime.isVisionExecutorError(msg),
            )
        }
    }

    @Test
    fun `isVisionExecutorError handles empty input`() {
        assertFalse(LiteRtRuntime.isVisionExecutorError(""))
    }

    // ---- argumentsToJson -------------------------------------------------------------
    //
    // The SDK hands tool-call arguments back as plain Kotlin values. This is what turns
    // them into the `input` string on UIMessagePart.Tool, so the host parses a local
    // model's arguments through exactly the same path as a cloud provider's.

    @Test
    fun `argumentsToJson preserves primitive types rather than stringifying them`() {
        val json = Json.parseToJsonElement(
            LiteRtRuntime.argumentsToJson(
                mapOf("s" to "hi", "n" to 42, "d" to 1.5, "b" to true)
            )
        ).jsonObject
        assertEquals("hi", json["s"]!!.jsonPrimitive.content)
        assertEquals("42", json["n"]!!.jsonPrimitive.content)
        assertFalse("numbers must not be quoted", json["n"]!!.jsonPrimitive.isString)
        assertEquals("1.5", json["d"]!!.jsonPrimitive.content)
        assertEquals(true, json["b"]!!.jsonPrimitive.content.toBoolean())
    }

    @Test
    fun `argumentsToJson walks nested objects and arrays`() {
        val json = Json.parseToJsonElement(
            LiteRtRuntime.argumentsToJson(
                mapOf(
                    "outer" to mapOf("inner" to listOf(1, "two", null)),
                )
            )
        ).jsonObject
        val inner = json["outer"]!!.jsonObject["inner"]!!.jsonArray
        assertEquals(3, inner.size)
        assertEquals("1", inner[0].jsonPrimitive.content)
        assertEquals("two", inner[1].jsonPrimitive.content)
        assertTrue(inner[2] is kotlinx.serialization.json.JsonNull)
    }

    @Test
    fun `argumentsToJson emits an empty object for a tool called with no arguments`() {
        assertEquals("{}", LiteRtRuntime.argumentsToJson(emptyMap()))
    }

    @Test
    fun `argumentsToJson keeps an unrecognised leaf as a string instead of dropping the key`() {
        val json = Json.parseToJsonElement(
            LiteRtRuntime.argumentsToJson(mapOf("weird" to StringBuilder("xyz")))
        ).jsonObject
        assertEquals("xyz", json["weird"]!!.jsonPrimitive.content)
    }
}
