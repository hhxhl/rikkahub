package me.rerere.rikkahub.data.ai.tools.local

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
sealed class LocalToolOption {
    @Serializable
    @SerialName("javascript_engine")
    data object JavascriptEngine : LocalToolOption()

    @Serializable
    @SerialName("time_info")
    data object TimeInfo : LocalToolOption()

    @Serializable
    @SerialName("clipboard")
    data object Clipboard : LocalToolOption()

    @Serializable
    @SerialName("tts")
    data object Tts : LocalToolOption()

    @Serializable
    @SerialName("ask_user")
    data object AskUser : LocalToolOption()

    @Serializable
    @SerialName("screen_time")
    data object ScreenTime : LocalToolOption()

    @Serializable
    @SerialName("calendar")
    data object Calendar : LocalToolOption()

    @Serializable
    @SerialName("chart_display")
    data object ChartDisplay : LocalToolOption()

    @Serializable
    @SerialName("device_volume")
    data object DeviceVolume : LocalToolOption()

    @Serializable
    @SerialName("device_brightness")
    data object DeviceBrightness : LocalToolOption()

    @Serializable
    @SerialName("device_torch")
    data object DeviceTorch : LocalToolOption()

    @Serializable
    @SerialName("device_battery")
    data object DeviceBattery : LocalToolOption()

    @Serializable
    @SerialName("device_audio_info")
    data object DeviceAudioInfo : LocalToolOption()

    @Serializable
    @SerialName("device_wifi_info")
    data object DeviceWifiInfo : LocalToolOption()

    @Serializable
    @SerialName("device_telephony_info")
    data object DeviceTelephonyInfo : LocalToolOption()

    @Serializable
    @SerialName("phone_use")
    data object PhoneUse : LocalToolOption()
}
