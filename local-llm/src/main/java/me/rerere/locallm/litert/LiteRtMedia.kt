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
import java.net.URL

/**
 * Adapter moved from Agent's ai/util/FileEncoder.kt, kept entirely inside the LiteRT module.
 * RikkaHub upstream does not expose Image.toBitmap / Audio.audioBytes; its cloud encoder
 * deliberately has a different contract. Do NOT change the upstream encoder to suit LiteRT.
 */
internal fun UIMessagePart.Image.toLiteRtBitmap(context: Context, maxDimension: Int = 1536): Bitmap? =
    runCatching {
        val uri = Uri.parse(url)
        val bytes = when (uri.scheme?.lowercase()) {
            "data" -> Base64.decode(url.substringAfter(',', ""), Base64.DEFAULT)
            "file" -> File(uri.path ?: return@runCatching null).readBytes()
            "content" -> context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            "http", "https" -> URL(url).openConnection().apply {
                connectTimeout = 8000
                readTimeout = 15000
            }.getInputStream().use { it.readBytes() }
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
        "data" -> Base64.decode(url.substringAfter(',', ""), Base64.DEFAULT)
        "file" -> File(uri.path ?: return@runCatching null).readBytes()
        "content" -> context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
        "http", "https" -> URL(url).openConnection().apply {
            connectTimeout = 8000
            readTimeout = 15000
        }.getInputStream().use { it.readBytes() }
        else -> null
    }
}.getOrNull()

private fun calculateSampleSize(width: Int, height: Int, maxDimension: Int): Int {
    var sample = 1
    while (width / sample > maxDimension || height / sample > maxDimension) sample *= 2
    return sample
}
