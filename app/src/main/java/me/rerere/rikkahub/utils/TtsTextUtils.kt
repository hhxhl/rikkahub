package me.rerere.rikkahub.utils

import android.icu.lang.UCharacter
import android.icu.lang.UProperty
import android.icu.text.BreakIterator
import java.util.Locale

/**
 * Replace each Emoji run with a natural Chinese comma only when it separates
 * two pieces of speech. Emoji at the edges of a sentence, or beside existing
 * punctuation, are removed without adding a redundant pause.
 *
 * Use Android ICU grapheme clusters to remove complete flags, keycaps,
 * skin-tone sequences and ZWJ-family Emoji. Chat content is not modified.
 */
fun String.removeEmojiForTts(): String {
    if (isEmpty()) return this

    val iterator = BreakIterator.getCharacterInstance(Locale.ROOT)
    iterator.setText(this)
    val spans = mutableListOf<TtsSpan>()
    var start = iterator.first()
    var end = iterator.next()
    while (end != BreakIterator.DONE) {
        if (isEmojiCluster(start, end)) {
            spans += TtsSpan(start, end)
        }
        start = end
        end = iterator.next()
    }
    return replaceSpeechSeparators(spans)
}

/**
 * Normalize ~, ～ and 〜 for speech, after the existing Markdown cleanup:
 * - Numeric ranges become "到": 10 ~ 20 -> 10到20.
 * - At sentence edges or beside punctuation: remove the tilde.
 * - Between spoken words: use one Chinese comma for a natural short pause.
 * The Emoji and tilde settings remain independent.
 */
fun String.normalizeTildeForTts(): String {
    val rangesExpanded = replace(TTS_NUMERIC_RANGE, "到")
    return rangesExpanded.replaceSpeechSeparators(
        TTS_TILDES.findAll(rangesExpanded).map { TtsSpan(it.range.first, it.range.last + 1) }.toList()
    )
}

private data class TtsSpan(val start: Int, val endExclusive: Int)

/**
 * Adjacent decorative symbols (including intervening spaces/tabs) count as
 * ONE separator. A comma is emitted only if useful text exists on both sides
 * and neither side is already punctuation or a sentence boundary.
 */
private fun String.replaceSpeechSeparators(spans: List<TtsSpan>): String {
    if (spans.isEmpty()) return this

    val result = StringBuilder(length)
    var sourceOffset = 0
    var index = 0
    while (index < spans.size) {
        val start = spans[index].start
        var end = spans[index].endExclusive
        while (index + 1 < spans.size &&
            substring(end, spans[index + 1].start).all { it.isHorizontalSpace() }
        ) {
            index++
            end = spans[index].endExclusive
        }

        result.append(this, sourceOffset, start)
        while (result.isNotEmpty() && result[result.length - 1].isHorizontalSpace()) {
            result.deleteCharAt(result.length - 1)
        }

        var nextOffset = end
        while (nextOffset < length && this[nextOffset].isHorizontalSpace()) {
            nextOffset++
        }

        val before = result.lastOrNull()
        val after = getOrNull(nextOffset)
        if (before != null && after != null &&
            !before.isSpeechPauseBoundary() && !after.isSpeechPauseBoundary()
        ) {
            result.append('，')
        }
        sourceOffset = nextOffset
        index++
    }
    result.append(this, sourceOffset, length)
    return result.toString().replace(DUPLICATE_HORIZONTAL_WHITESPACE, " ").trim()
}

private fun Char.isHorizontalSpace() = this == ' ' || this == '\t'

private fun Char.isSpeechPauseBoundary(): Boolean {
    if (this == '\n' || this == '\r' || this.isWhitespace()) return true
    return when (Character.getType(this)) {
        Character.CONNECTOR_PUNCTUATION.toInt(),
        Character.DASH_PUNCTUATION.toInt(),
        Character.START_PUNCTUATION.toInt(),
        Character.END_PUNCTUATION.toInt(),
        Character.INITIAL_QUOTE_PUNCTUATION.toInt(),
        Character.FINAL_QUOTE_PUNCTUATION.toInt(),
        Character.OTHER_PUNCTUATION.toInt() -> true
        else -> false
    }
}

private val TTS_NUMERIC_RANGE = Regex("""(?<=\p{Nd})[ \t]*[~～〜]+[ \t]*(?=\p{Nd})""")
private val TTS_TILDES = Regex("""[~～〜]+""")
private val DUPLICATE_HORIZONTAL_WHITESPACE = Regex("[ \t]{2,}")

private fun String.isEmojiCluster(start: Int, end: Int): Boolean {
    var offset = start
    var hasTextStyleSymbol = false
    var hasEmojiVariationSelector = false
    while (offset < end) {
        val codePoint = Character.codePointAt(this, offset)
        when (codePoint) {
            0xFE0F -> hasEmojiVariationSelector = true
            // Copyright/trademark symbols have a non-Emoji text presentation by default.
            0x00A9, 0x00AE, 0x2122 -> hasTextStyleSymbol = true
            else -> if (codePoint == 0x20E3 || // keycap: 1️⃣
                codePoint in 0x1F1E6..0x1F1FF || // flag sequences
                UCharacter.hasBinaryProperty(codePoint, UProperty.EMOJI_PRESENTATION) ||
                UCharacter.hasBinaryProperty(codePoint, UProperty.EXTENDED_PICTOGRAPHIC)
            ) {
                return true
            }
        }
        offset += Character.charCount(codePoint)
    }
    return hasTextStyleSymbol && hasEmojiVariationSelector
}
