package me.rerere.locallm.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import me.rerere.ai.core.ReasoningLevel
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Pure-decision-function tests for [LiteRtProvider]'s image-forwarding gate.
 *
 * [decideImageForwarding] guards against a hard native crash. When a vision-capable model is
 * loaded but the device's GPU vision executor fails to initialise, [LiteRtRuntime.ensureLoaded]
 * falls back to a TEXT-ONLY engine (see [LiteRtRuntime.LoadOutcome.visionEnabled] /
 * [LiteRtRuntime.LoadOutcome.visionFellBackToTextOnly]). Forwarding image bytes to an engine
 * that has no vision executor null-derefs inside `liblitertlm_jni.so` -> SIGSEGV. The gate
 * MUST therefore key off the ACTUAL post-load vision state, never the pre-load estimate (which
 * is still `true` at the moment of the doomed first load).
 */
class LiteRtProviderTest {

    @Test
    fun `OFF disables native thinking without modifying tool availability`() {
        val config = liteRtThinkingConfig(ReasoningLevel.OFF)!!
        assertFalse(config.enableThinking)
        assertEquals(0, config.thinkingTokenBudget)
    }

    @Test
    fun `AUTO inherits model thinking defaults`() {
        assertNull(liteRtThinkingConfig(ReasoningLevel.AUTO))
    }

    @Test
    fun `explicit reasoning levels use RikkaHub native token budgets`() {
        for (level in listOf(ReasoningLevel.LOW, ReasoningLevel.MEDIUM,
            ReasoningLevel.HIGH, ReasoningLevel.XHIGH, ReasoningLevel.MAX)) {
            val config = liteRtThinkingConfig(level)!!
            assertTrue(config.enableThinking)
            assertEquals(level.budgetTokens, config.thinkingTokenBudget)
        }
    }

    @Test
    fun `forwards images when vision is live post-load`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = true,
            userSentImages = true,
        )
        assertTrue("vision live -> forward", d.forwardImages)
        assertFalse("nothing dropped when we forward", d.noteImagesDropped)
    }

    @Test
    fun `does NOT forward images when vision fell back to text-only (the crash case)`() {
        // Vision-capable model, but the GPU vision executor failed and the engine loaded
        // text-only. Forwarding here is the SIGSEGV. Drop the images, and flag a user note.
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = false,
            userSentImages = true,
        )
        assertFalse("vision not live -> never forward", d.forwardImages)
        assertTrue("user attached images that were dropped -> note them", d.noteImagesDropped)
    }

    @Test
    fun `no note when vision fell back but the user attached no images`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = false,
            userSentImages = false,
        )
        assertFalse(d.forwardImages)
        assertFalse("nothing to note when no images were attached", d.noteImagesDropped)
    }

    @Test
    fun `text-only model with a stray image attachment drops silently (no note)`() {
        // The user picked a model that never supported vision. Preserve the prior silent-drop
        // behaviour: no "vision unavailable on this device" note, because the device's vision
        // capability was never the issue here.
        val d = decideImageForwarding(
            modelImageCapable = false,
            visionEnabledPostLoad = false,
            userSentImages = true,
        )
        assertFalse(d.forwardImages)
        assertFalse("text-only model is not a device-vision failure", d.noteImagesDropped)
    }

    @Test
    fun `no images with vision live is a no-op`() {
        val d = decideImageForwarding(
            modelImageCapable = true,
            visionEnabledPostLoad = true,
            userSentImages = false,
        )
        assertTrue(d.forwardImages)
        assertFalse(d.noteImagesDropped)
    }
}

/**
 * Budget tests for the native tool declarations.
 *
 * Tool declarations are prompt text: the chat template renders each one ahead of the
 * conversation. They were the one part of the prefill with no cap, so an assistant with a
 * large enabled tool set pushed tens of thousands of characters of JSON schema into a model
 * whose whole context is a few thousand tokens. The engine does not bounds-check that: it
 * faults inside the native executor rather than returning a context-overflow error.
 */
class LiteRtToolDeclarationBudgetTest {

    @Test
    fun `input budget grows with actual Engine context`() {
        assertTrue(LiteRtProvider.promptByteBudget(8192) > LiteRtProvider.promptByteBudget(4096))
        assertTrue(LiteRtProvider.promptByteBudget(16384) > LiteRtProvider.promptByteBudget(8192))
    }

    @Test
    fun `small 1280-token models reserve space for dialogue rather than tool schemas`() {
        assertEquals(0, LiteRtProvider.toolDeclarationByteBudget(1280, 500))
        assertTrue(LiteRtProvider.historyByteBudget(1280, 500, 0) > 0)
    }

    @Test
    fun `default threshold still blocks native tools at 1280 tokens`() {
        assertEquals(0, LiteRtProvider.toolDeclarationByteBudget(1280, 500, 25, 1280))
        assertTrue(LiteRtProvider.toolDeclarationByteBudget(1281, 500, 25, 1280) > 0)
    }

    @Test
    fun `zero threshold allows tools when declaration budget has space`() {
        val budget = LiteRtProvider.toolDeclarationByteBudget(1280, 500, 25, 0)
        assertEquals(614, budget)
        assertEquals(0, LiteRtProvider.toolDeclarationByteBudget(1280, 2000, 25, 0))
    }

    @Test
    fun `custom threshold suppresses tools even above former 1280 cutoff`() {
        assertEquals(0, LiteRtProvider.toolDeclarationByteBudget(2048, 500, 25, 2048))
        assertTrue(LiteRtProvider.toolDeclarationByteBudget(2049, 500, 25, 2048) > 0)
    }

    @Test
    fun `tool and history budgets stay within current Engine allocation`() {
        val budget = LiteRtProvider.promptByteBudget(8192)
        val system = 780
        val tool = LiteRtProvider.toolDeclarationByteBudget(8192, system)
        val history = LiteRtProvider.historyByteBudget(8192, system, tool)
        assertEquals(budget, system + tool + history)
        assertTrue(history > 3000)
    }

    @Test
    fun `enabled tools reduce allowed history without a fixed 3000-character cap`() {
        val freeHistory = LiteRtProvider.historyByteBudget(16384, 500, 0)
        val withTools = LiteRtProvider.historyByteBudget(16384, 500, 4000)
        assertTrue(freeHistory > 3000)
        assertEquals(freeHistory - 4000, withTools)
    }
}

/**
 * The prompt has to be budgeted against the context the ENGINE was configured with, never
 * against the ceiling the model file could theoretically hold. Those are different numbers
 * whenever the catalog allocates less than the file supports, and budgeting against the
 * ceiling let a 32768-token budget build a prompt for an engine holding 4096: on
 * gemma-4-E2B the whole enabled tool set was declared, 31067 chars of it, and the engine
 * rejected the prefill at 7868 tokens.
 */
class LiteRtEngineContextTokensTest {

    @Test
    fun `automatic capacity uses known ceiling but unknown capacity stays conservative`() {
        val config = LiteRtModelDefaults.forModelFile("unknown.litertlm")
        assertEquals(8192, LiteRtProvider.configuredContextTokens(config, 8192, null))
        assertEquals(2048, LiteRtProvider.configuredContextTokens(config, 8192, 2048))
        assertEquals(4096, LiteRtProvider.configuredContextTokens(config, null, 32768))
        assertEquals(32768, LiteRtProvider.configuredContextTokens(config, 65536, null))
    }

    @Test
    fun `the allocation wins when the file ceiling is higher`() {
        // gemma-4-E2B: catalog allocates 4096, the file could hold 32768.
        assertEquals(4096, LiteRtProvider.engineContextTokens(4096, 32768))
    }

    @Test
    fun `the ceiling wins when the request is higher`() {
        // qwen3_0_6b_mixed_int4 declares 2048; asking for more faults the native executor.
        assertEquals(2048, LiteRtProvider.engineContextTokens(32000, 2048))
    }

    @Test
    fun `an unknown ceiling leaves the request untouched`() {
        assertEquals(4096, LiteRtProvider.engineContextTokens(4096, null))
    }

    @Test
    fun `the tool budget derived from the engine size never exceeds the engine`() {
        // The regression this guards: budget(ceiling) was handed to an engine sized at the
        // allocation, so the committed prefill could run past what the engine can hold.
        val engine = LiteRtProvider.engineContextTokens(4096, 32768)
        val systemBytes = 500
        val toolBytes = LiteRtProvider.toolDeclarationByteBudget(engine, systemBytes)
        val historyBytes = LiteRtProvider.historyByteBudget(engine, systemBytes, toolBytes)
        assertEquals(LiteRtProvider.promptByteBudget(engine),
            systemBytes + toolBytes + historyBytes)
    }
}

/** The message cap is user-configurable, rather than always hard-coded to 500. */
class LiteRtSystemMessageLimitTest {
    @Test
    fun `500-character default truncates a longer system message`() {
        val text = "a".repeat(700)
        val result = truncateLocalSystemText(text, 500)
        assertTrue(result.startsWith("a".repeat(500)))
        assertTrue(result.contains("truncated for local model context"))
    }

    @Test
    fun `larger user limit retains the same system message intact`() {
        val text = "a".repeat(700)
        assertEquals(text, truncateLocalSystemText(text, 1000))
    }

    @Test
    fun `short system message is unchanged`() {
        assertEquals("你好", truncateLocalSystemText("你好", 500))
    }
}

/** Verify the configured output reserve is applied to system, tool and history budgets. */
class LiteRtResponseReserveBudgetTest {
    @Test
    fun `25 percent preserves the previous 1280-token default`() {
        assertEquals(1728, LiteRtProvider.promptByteBudget(1280, 25))
        assertEquals(LiteRtProvider.promptByteBudget(1280), LiteRtProvider.promptByteBudget(1280, 25))
    }

    @Test
    fun `larger reserve leaves less room for input and history`() {
        val defaultBudget = LiteRtProvider.promptByteBudget(1280, 25)
        val largerReserve = LiteRtProvider.promptByteBudget(1280, 40)
        assertEquals(1344, largerReserve)
        assertTrue(largerReserve < defaultBudget)
        assertTrue(LiteRtProvider.historyByteBudget(1280, 300, 0, 40) <
            LiteRtProvider.historyByteBudget(1280, 300, 0, 25))
    }

    @Test
    fun `smaller reserve still protects minimum 256 output tokens`() {
        assertEquals(1856, LiteRtProvider.promptByteBudget(1280, 10))
        assertEquals(1856, LiteRtProvider.promptByteBudget(1280, 20))
    }

    @Test
    fun `tool budgets also honor the configured reserve`() {
        assertTrue(LiteRtProvider.toolDeclarationByteBudget(4096, 300, 40) <
            LiteRtProvider.toolDeclarationByteBudget(4096, 300, 25))
    }
}

/** History trimming must preserve USER-led boundaries, not orphan earlier assistant turns. */
class LiteRtHistoryTrimmingTest {
    @Test
    fun `drops a full old question and answer`() {
        val entries = listOf("U:first", "A:old", "U:second", "A:new", "U:third")
        assertEquals(
            listOf("U:second", "A:new", "U:third"),
            trimOldestConversationSegment(entries) { it.startsWith("U:") },
        )
    }

    @Test
    fun `keeps latest user even when time reminder is a separate user message`() {
        val entries = listOf("U:old", "A:old", "U:time reminder", "U:question")
        val once = trimOldestConversationSegment(entries) { it.startsWith("U:") }
        assertEquals(listOf("U:time reminder", "U:question"), once)
        assertEquals(
            listOf("U:question"),
            trimOldestConversationSegment(once) { it.startsWith("U:") },
        )
    }

    @Test
    fun `will not orphan an assistant when no later user exists`() {
        val entries = listOf("U:question", "A:tool-result")
        assertEquals(entries, trimOldestConversationSegment(entries) { it.startsWith("U:") })
    }
}
