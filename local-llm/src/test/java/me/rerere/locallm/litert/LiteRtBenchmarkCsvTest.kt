package me.rerere.locallm.litert

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LiteRtBenchmarkCsvTest {
    private fun info(init: Double, prefill: Double, decode: Double): LiteRtBenchmarkRunner.Metrics = LiteRtBenchmarkRunner.Metrics(
        initTimeInSecond = init,
        timeToFirstTokenInSecond = 0.5,
        lastPrefillTokenCount = 256,
        lastDecodeTokenCount = 256,
        lastPrefillTokensPerSecond = prefill,
        lastDecodeTokensPerSecond = decode,
    )

    @Test
    fun `summarizes original native benchmark values without estimates`() {
        val result = LiteRtBenchmarkRunner.Result(
            modelName = "Gemma 3 1B-IT", accelerator = "CPU",
            prefillTokens = 256, decodeTokens = 256,
            startedAtMs = 10, finishedAtMs = 20,
            runs = listOf(
                LiteRtBenchmarkRunner.Run(1, info(12.0, 120.0, 18.0)),
                LiteRtBenchmarkRunner.Run(2, info(4.0, 140.0, 20.0)),
                LiteRtBenchmarkRunner.Run(3, info(6.0, 160.0, 22.0)),
            ),
        )
        assertEquals(140.0, result.prefillTps, 0.001)
        assertEquals(20.0, result.decodeTps, 0.001)
        assertEquals(12000.0, result.firstInitMs, 0.001)
        assertEquals(5000.0, result.steadyInitMs!!, 0.001)
        assertEquals(13, result.toCsv("3.2.0").lines().first().split(',').size)
        assertTrue(result.toCsv("3.2.0").contains("\"Gemma 3 1B-IT\""))
    }

    @Test
    fun `escapes user supplied model filenames in csv`() {
        val result = LiteRtBenchmarkRunner.Result(
            modelName = "=SUM(1,2)\".litertlm", accelerator = "GPU",
            prefillTokens = 256, decodeTokens = 256,
            startedAtMs = 10, finishedAtMs = 20,
            runs = listOf(LiteRtBenchmarkRunner.Run(1, info(2.0, 50.0, 12.0))),
        )
        val csv = result.toCsv("3.2.0")
        assertTrue(csv.contains("\"'=SUM(1,2)\"\".litertlm\""))
        assertTrue(csv.lines()[1].endsWith(",")) // no steady init for one run
    }
}
