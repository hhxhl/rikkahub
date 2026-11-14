package me.rerere.asr

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ASRProviderSettingAndroidSystemTest {
    @Test
    fun android_system_is_available_for_voice_mode() {
        val setting = ASRProviderSetting.AndroidSystem()
        assertEquals("Android System ASR", setting.name)
        assertEquals("zh-CN", setting.language)
        assertTrue(setting.supportsServerVadVoiceMode)
        assertTrue(ASRProviderSetting.Types.contains(ASRProviderSetting.AndroidSystem::class))
    }

    @Test
    fun android_system_copy_provider_keeps_language() {
        val setting = ASRProviderSetting.AndroidSystem(language = "en-US")
        val renamed = setting.copyProvider(id = setting.id, name = "Local ASR")
        assertTrue(renamed is ASRProviderSetting.AndroidSystem)
        assertEquals("Local ASR", renamed.name)
        assertEquals("en-US", (renamed as ASRProviderSetting.AndroidSystem).language)
    }

    @Test
    fun one_shot_http_services_are_still_excluded_from_voice_mode() {
        assertFalse(ASRProviderSetting.MiMo().supportsServerVadVoiceMode)
        assertFalse(ASRProviderSetting.Step().supportsServerVadVoiceMode)
    }
}
