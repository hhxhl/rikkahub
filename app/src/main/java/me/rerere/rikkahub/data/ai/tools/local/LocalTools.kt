package me.rerere.rikkahub.data.ai.tools.local

import android.content.Context
import me.rerere.ai.core.Tool
import me.rerere.rikkahub.data.datastore.SettingsStore
import me.rerere.rikkahub.data.event.AppEventBus
import me.rerere.rikkahub.data.files.FilesManager
import me.rerere.rikkahub.phoneuse.PhoneUseManager
import me.rerere.rikkahub.phoneuse.isFloatingChatWindowOn
import me.rerere.rikkahub.phoneuse.isPhoneUseCheckedInSystem
import me.rerere.tts.provider.TTSManager

class LocalTools(
    private val context: Context,
    private val eventBus: AppEventBus,
    private val ttsManager: TTSManager,
    private val settingsStore: SettingsStore,
    private val phoneUseManager: PhoneUseManager,
    private val filesManager: FilesManager,
) {
    val javascriptTool by lazy { buildJavascriptTool() }

    val timeTool by lazy { buildTimeInfoTool() }

    val clipboardTool by lazy { buildClipboardTool(context) }

    val ttsTool by lazy { buildTextToSpeechTool(eventBus, ttsManager, settingsStore) }

    val askUserTool by lazy { buildAskUserTool() }

    val screenTimeTool by lazy { buildScreenTimeTool(context, eventBus) }

    val calendarQueryTool by lazy { buildCalendarQueryTool(context) }

    val calendarCreateTool by lazy { buildCalendarCreateTool(context) }

    val chartDisplayTool by lazy { buildChartDisplayTool() }

    private val deviceVolumeTool by lazy { buildDeviceTool(context, "volume") }
    private val deviceBrightnessTool by lazy { buildDeviceTool(context, "brightness") }
    private val deviceTorchTool by lazy { buildDeviceTool(context, "torch") }
    private val deviceBatteryTool by lazy { buildDeviceTool(context, "battery") }
    private val deviceAudioInfoTool by lazy { buildDeviceTool(context, "audio_info") }
    private val deviceWifiInfoTool by lazy { buildDeviceTool(context, "wifi_info") }
    private val deviceTelephonyInfoTool by lazy { buildDeviceTool(context, "telephony_info") }

    val phoneUseTool by lazy {
        buildPhoneUseTool(
            ownPackageName = context.packageName,
            manager = phoneUseManager,
            isFloatingWindowOn = { isFloatingChatWindowOn(context, settingsStore.settingsFlow.value) },
            isCheckedInSystem = { isPhoneUseCheckedInSystem(context) },
            saveScreenshot = { bytes ->
                filesManager.createChatFilesByByteArrays(
                    byteArrays = listOf(bytes),
                    displayName = "screenshot.jpg",
                    mimeType = "image/jpeg",
                ).first().toString()
            },
        )
    }

    fun getTools(options: List<LocalToolOption>): List<Tool> {
        val tools = mutableListOf<Tool>()
        if (options.contains(LocalToolOption.JavascriptEngine)) {
            tools.add(javascriptTool)
        }
        if (options.contains(LocalToolOption.TimeInfo)) {
            tools.add(timeTool)
        }
        if (options.contains(LocalToolOption.Clipboard)) {
            tools.add(clipboardTool)
        }
        if (options.contains(LocalToolOption.Tts)) {
            tools.add(ttsTool)
        }
        if (options.contains(LocalToolOption.AskUser)) {
            tools.add(askUserTool)
        }
        if (options.contains(LocalToolOption.ScreenTime)) {
            tools.add(screenTimeTool)
        }
        if (options.contains(LocalToolOption.Calendar)) {
            tools.add(calendarQueryTool)
            tools.add(calendarCreateTool)
        }
        if (options.contains(LocalToolOption.ChartDisplay)) {
            tools.add(chartDisplayTool)
        }
        if (options.contains(LocalToolOption.DeviceVolume)) tools.add(deviceVolumeTool)
        if (options.contains(LocalToolOption.DeviceBrightness)) tools.add(deviceBrightnessTool)
        if (options.contains(LocalToolOption.DeviceTorch)) tools.add(deviceTorchTool)
        if (options.contains(LocalToolOption.DeviceBattery)) tools.add(deviceBatteryTool)
        if (options.contains(LocalToolOption.DeviceAudioInfo)) tools.add(deviceAudioInfoTool)
        if (options.contains(LocalToolOption.DeviceWifiInfo)) tools.add(deviceWifiInfoTool)
        if (options.contains(LocalToolOption.DeviceTelephonyInfo)) tools.add(deviceTelephonyInfoTool)
        // 不支持的渠道和系统版本上，之前在别处开过（比如导入的备份）也不提供
        if (options.contains(LocalToolOption.PhoneUse) && PhoneUseManager.isSupported) {
            tools.add(phoneUseTool)
        }
        return tools
    }
}
