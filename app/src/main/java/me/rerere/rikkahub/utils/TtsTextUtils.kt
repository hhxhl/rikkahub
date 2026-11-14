package me.rerere.rikkahub.utils

import android.icu.lang.UCharacter
import android.icu.lang.UProperty
import android.icu.text.BreakIterator
import java.util.Locale

/**
 * Replace Emoji in speech-only text with spaces without altering the chat message.
 *
 * Work on Unicode grapheme clusters rather than UTF-16 chars: this drops complete
 * skin-tone, ZWJ-family, flag and keycap Emoji without leaving invisible joiners.
 * Non-Emoji text and punctuation are retained; repeated spaces are collapsed.
 * Android ICU is available from minSdk 26 (the app targets minSdk 26).
 */
fun String.removeEmojiForTts(): String {
    if (isEmpty()) return this

    val iterator = BreakIterator.getCharacterInstance(Locale.ROOT)
    iterator.setText(this)
    return buildString(length) {
        var start = iterator.first()
        var end = iterator.next()
        while (end != BreakIterator.DONE) {
            if (this@removeEmojiForTts.isEmojiCluster(start, end)) {
                // Replace each Emoji cluster with a space to avoid joining spoken words.
                append(' ')
            } else {
                append(this@removeEmojiForTts, start, end)
            }
            start = end
            end = iterator.next()
        }
    }.replace(DUPLICATE_HORIZONTAL_WHITESPACE, " ").trim()
}

/**
 * Normalize spoken tilde characters after Markdown cleanup.
 * - A numeric range (including spaces around the tilde) becomes "到": 10~20 -> 10到20.
 * - A tilde at the start/end of a sentence or line is removed.
 * - Tildes between other text become spaces, avoiding glued-together words.
 * Applies to ASCII ~, fullwidth ～ and wave dash 〜.
 */
fun String.normalizeTildeForTts(): String =
    replace(TTS_NUMERIC_RANGE, "到")
        .replace(TTS_LEADING_TILDE, "$1")
        .replace(TTS_TRAILING_TILDE, "")
        .replace(TTS_OTHER_TILDES, " ")
        .replace(DUPLICATE_HORIZONTAL_WHITESPACE, " ")
        .trim()

private val TTS_NUMERIC_RANGE = Regex("""(?<=\p{Nd})[ \t]*[~～〜]+[ \t]*(?=\p{Nd})""")
private val TTS_LEADING_TILDE = Regex("""(^|[。！？!?；;.\n])[ \t]*[~～〜]+[ \t]*""")
private val TTS_TRAILING_TILDE = Regex("""[ \t]*[~～〜]+[ \t]*(?=$|[。！？!?；;.\n])""")
private val TTS_OTHER_TILDES = Regex("""[~～〜]+""")
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
