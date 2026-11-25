package me.rerere.locallm.litert

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.ExifInterface
import android.util.Base64
import android.net.Uri
import me.rerere.ai.ui.UIMessagePart
import java.io.File
import java.io.InputStream
import java.io.ByteArrayOutputStream
import java.net.URL

private const val MAX_ATTACHMENT_BYTES = 16 * 1024 * 1024

/** Bounded input even for content/HTTP streams with no trustworthy length header. */
internal fun InputStream.readLiteRtBounded(maxBytes: Int = MAX_ATTACHMENT_BYTES): ByteArray {
    require(maxBytes > 0)
    val out = ByteArrayOutputStream(minOf(8192, maxBytes))
    val buffer = ByteArray(8192)
    var total = 0
    while (true) {
        val read = read(buffer)
        if (read < 0) break
        if (read == 0) continue
        require(read <= maxBytes - total) { "LiteRT attachment exceeds $maxBytes bytes" }
        out.write(buffer, 0, read)
        total += read
    }
    return out.toByteArray()
}

private fun decodeBoundedDataUrl(url: String): ByteArray {
    val base64 = url.substringAfter(',', "")
    require(base64.length.toLong() <= ((MAX_ATTACHMENT_BYTES.toLong() + 2) / 3) * 4 + 4096) {
        "LiteRT attachment exceeds encoded size limit"
    }
    return Base64.decode(base64, Base64.DEFAULT).also {
        require(it.size <= MAX_ATTACHMENT_BYTES) { "LiteRT attachment exceeds size limit" }
    }
}

/**
 * Adapter moved from Agent's ai/util/FileEncoder.kt, kept entirely inside the LiteRT module.
 * RikkaHub upstream does not expose Image.toBitmap / Audio.audioBytes; its cloud encoder
 * deliberately has a different contract. Do NOT change the upstream encoder to suit LiteRT.
 */
internal fun UIMessagePart.Image.toLiteRtBitmap(context: Context, maxDimension: Int = 1536): Bitmap? =
    runCatching {
        val uri = Uri.parse(url)
        val bytes = when (uri.scheme?.lowercase()) {
            "data" -> decodeBoundedDataUrl(url)
            "file" -> File(uri.path ?: return@runCatching null).inputStream().use { it.readLiteRtBounded() }
            "content" -> context.contentResolver.openInputStream(uri)?.use { it.readLiteRtBounded() }
            "http", "https" -> URL(url).openConnection().apply {
                connectTimeout = 8000
                readTimeout = 15000
            }.getInputStream().use { it.readLiteRtBounded() }
            else -> null
        } ?: return@runCatching null
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return@runCatching null
        val sample = calculateSampleSize(bounds.outWidth, bounds.outHeight, maxDimension)
        val options = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, options)
            ?: return@runCatching null
        if (uri.scheme != "file") return@runCatching decoded
        val path = uri.path ?: return@runCatching decoded
        val orientation = ExifInterface(path).getAttributeInt(
            ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL,
        )
        val angle = when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90f
            ExifInterface.ORIENTATION_ROTATE_180 -> 180f
            ExifInterface.ORIENTATION_ROTATE_270 -> 270f
            else -> 0f
        }
        if (angle == 0f) decoded else Bitmap.createBitmap(
            decoded, 0, 0, decoded.width, decoded.height,
            Matrix().apply { postRotate(angle) }, true,
        ).also { decoded.recycle() }
    }.getOrNull()

internal fun UIMessagePart.Audio.toLiteRtAudioBytes(context: Context): ByteArray? = runCatching {
    val uri = Uri.parse(url)
    when (uri.scheme?.lowercase()) {
        "data" -> decodeBoundedDataUrl(url)
        "file" -> File(uri.path ?: return@runCatching null).inputStream().use { it.readLiteRtBounded() }
        "content" -> context.contentResolver.openInputStream(uri)?.use { it.readLiteRtBounded() }
        "http", "https" -> URL(url).openConnection().apply {
            connectTimeout = 8000
            readTimeout = 15000
        }.getInputStream().use { it.readLiteRtBounded() }
        else -> null
    }
}.getOrNull()

private fun calculateSampleSize(width: Int, height: Int, maxDimension: Int): Int {
    var sample = 1
    while (width / sample > maxDimension || height / sample > maxDimension) sample *= 2
    return sample
}
