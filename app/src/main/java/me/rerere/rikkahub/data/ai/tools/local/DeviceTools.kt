package me.rerere.rikkahub.data.ai.tools.local

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.app.NotificationManager
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraCharacteristics
import android.media.AudioManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.wifi.WifiManager
import android.os.BatteryManager
import android.provider.Settings
import android.telephony.TelephonyManager
import kotlinx.serialization.json.*
import me.rerere.ai.core.InputSchema
import me.rerere.ai.core.Tool
import me.rerere.ai.ui.UIMessagePart
import kotlin.math.roundToInt

internal fun buildDeviceTool(context: Context, kind: String): Tool = Tool(
    name = "device_$kind",
    description = when (kind) {
        "volume" -> "Get or set device volume percentage. Omit percent to query."
        "brightness" -> "Get or set screen brightness percentage. Setting disables automatic brightness."
        "torch" -> "Turn flashlight on or off."
        "battery" -> "Get battery percentage and charging status."
        "audio_info" -> "Get connected audio output devices and ringer mode."
        "wifi_info" -> "Get Wi-Fi connection status and network name when permitted."
        else -> "Get mobile network connection status and operator when permitted."
    },
    parameters = {
        InputSchema.Obj(properties = buildJsonObject {
            if (kind == "volume" || kind == "brightness") {
                put("percent", buildJsonObject {
                    put("type", "integer")
                    put("description", "0 to 100; omit to query")
                })
            }
            if (kind == "volume") put("stream", buildJsonObject {
                put("type", "string")
                put("enum", buildJsonArray { listOf("media", "ring", "notification", "alarm", "voice_call", "system").forEach { add(it) } })
                put("description", "Default media")
            })
            if (kind == "torch") put("on", buildJsonObject { put("type", "boolean") })
        }, required = if (kind == "torch") listOf("on") else emptyList())
    },
    execute = { args ->
        val result = try {
            val params = args.jsonObject
            val percent = params["percent"]?.jsonPrimitive?.intOrNull
            require("percent" !in params || percent != null && percent in 0..100) { "percent must be an integer from 0 to 100" }
            buildJsonObject {
                when (kind) {
                    "volume" -> {
                        val am = context.getSystemService(AudioManager::class.java)
                        val stream = params["stream"]?.jsonPrimitive?.contentOrNull ?: "media"
                        val id = when (stream) {
                            "media" -> AudioManager.STREAM_MUSIC
                            "ring" -> AudioManager.STREAM_RING
                            "notification" -> AudioManager.STREAM_NOTIFICATION
                            "alarm" -> AudioManager.STREAM_ALARM
                            "voice_call" -> AudioManager.STREAM_VOICE_CALL
                            "system" -> AudioManager.STREAM_SYSTEM
                            else -> error("Unknown stream")
                        }
                        val max = am.getStreamMaxVolume(id)
                        val min = if (android.os.Build.VERSION.SDK_INT >= 28) am.getStreamMinVolume(id) else 0
                        if (percent != null) {
                            if (stream in listOf("ring", "notification")) {
                                check(context.getSystemService(NotificationManager::class.java).isNotificationPolicyAccessGranted) { "Grant Do Not Disturb access in Android settings" }
                            }
                            am.setStreamVolume(id, (max * percent / 100f).roundToInt().coerceIn(min, max), 0)
                        }
                        put("stream", stream)
                        put("percent", if (max > 0) (am.getStreamVolume(id) * 100f / max).roundToInt() else 0)
                    }
                    "brightness" -> {
                        val cr = context.contentResolver
                        if (percent != null) {
                            check(Settings.System.canWrite(context)) { "Grant Modify system settings access in Android settings" }
                            check(Settings.System.putInt(cr, Settings.System.SCREEN_BRIGHTNESS_MODE, Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL))
                            check(Settings.System.putInt(cr, Settings.System.SCREEN_BRIGHTNESS, (percent * 255f / 100).roundToInt().coerceIn(1, 255)))
                        }
                        put("percent", (Settings.System.getInt(cr, Settings.System.SCREEN_BRIGHTNESS) * 100f / 255).roundToInt())
                        put("automatic", Settings.System.getInt(cr, Settings.System.SCREEN_BRIGHTNESS_MODE) == Settings.System.SCREEN_BRIGHTNESS_MODE_AUTOMATIC)
                    }
                    "torch" -> {
                        val on = params["on"]?.jsonPrimitive?.booleanOrNull ?: error("on is required")
                        val cm = context.getSystemService(CameraManager::class.java)
                        val id = cm.cameraIdList.firstOrNull { cm.getCameraCharacteristics(it).get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true } ?: error("No flashlight")
                        cm.setTorchMode(id, on)
                        put("on", on)
                    }
                    "battery" -> {
                        val i = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: error("Battery unavailable")
                        val level = i.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                        val scale = i.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                        if (level >= 0 && scale > 0) put("percent", level * 100 / scale)
                        val status = i.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
                        put("charging", status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL)
                        put("temperature_c", i.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) / 10.0)
                    }
                    "audio_info" -> {
                        val am = context.getSystemService(AudioManager::class.java)
                        put("ringer_mode", when (am.ringerMode) { AudioManager.RINGER_MODE_SILENT -> "silent"; AudioManager.RINGER_MODE_VIBRATE -> "vibrate"; else -> "normal" })
                        put("outputs", buildJsonArray {
                            am.getDevices(AudioManager.GET_DEVICES_OUTPUTS).forEach { d -> add(buildJsonObject { put("name", d.productName.toString()); put("type", d.type) }) }
                        })
                    }
                    "wifi_info" -> {
                        val cm = context.getSystemService(ConnectivityManager::class.java)
                        put("connected", cm.allNetworks.any { cm.getNetworkCapabilities(it)?.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) == true })
                        val wm = context.applicationContext.getSystemService(WifiManager::class.java)
                        put("enabled", wm.isWifiEnabled)
                        @Suppress("DEPRECATION")
                        val info = wm.connectionInfo
                        if (info.ssid != "<unknown ssid>") put("ssid", info.ssid)
                    }
                    "telephony_info" -> {
                        val cm = context.getSystemService(ConnectivityManager::class.java)
                        put("connected", cm.allNetworks.any { cm.getNetworkCapabilities(it)?.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) == true })
                        val tm = context.getSystemService(TelephonyManager::class.java)
                        put("operator", tm.networkOperatorName)
                        put("roaming", tm.isNetworkRoaming)
                    }
                }
            }
        } catch (e: Exception) {
            buildJsonObject { put("error", e.message ?: e.javaClass.simpleName) }
        }
        listOf(UIMessagePart.Text(result.toString()))
    }
)
