package me.rerere.rikkahub.ui.pages.setting.components

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import me.rerere.ai.provider.Model
import me.rerere.ai.provider.ProviderSetting
import me.rerere.locallm.LocalRuntime
import me.rerere.locallm.LocalRuntimePreferences
import me.rerere.locallm.ModelInstall
import me.rerere.locallm.litert.LiteRtCatalog
import me.rerere.locallm.litert.LiteRtModelDefaults
import me.rerere.locallm.litert.LiteRtModelMetadata
import me.rerere.locallm.litert.LiteRtRuntime
import me.rerere.rikkahub.data.datastore.SettingsStore
import okhttp3.OkHttpClient
import org.koin.compose.koinInject
import java.io.File

/**
 * Thin Compose adapter for the Agent's self-contained local-llm module.
 *
 * Keeps the original RikkaHub Provider UI, navigation, and model editor untouched.
 * The provider setting remains the canonical chat-model list; installed-model files,
 * CPU/GPU preference, and token limit live in Agent's LocalRuntimePreferences.
 */
@Composable
fun ProviderConfigureLiteRT(
    provider: ProviderSetting.LiteRtLocal,
    onEdit: (ProviderSetting.LiteRtLocal) -> Unit,
) {
    val context = LocalContext.current
    val prefs = koinInject<LocalRuntimePreferences>()
    val runtime = koinInject<LiteRtRuntime>()
    val settingsStore = koinInject<SettingsStore>()
    val client = koinInject<OkHttpClient>()
    val scope = rememberCoroutineScope()
    val currentProvider by rememberUpdatedState(provider)
    val currentOnEdit by rememberUpdatedState(onEdit)

    val installed by prefs.installedModelsFlow(LocalRuntime.LiteRT).collectAsState(initial = emptyMap())
    val forceCpu by prefs.forceCpuFlow(LocalRuntime.LiteRT).collectAsState(initial = true)
    val contextOverride by prefs.maxNumTokensOverrideFlow(LocalRuntime.LiteRT).collectAsState(initial = null)
    var tokenField by remember(contextOverride) { mutableStateOf(contextOverride?.toString().orEmpty()) }
    var url by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var status by remember { mutableStateOf<String?>(null) }
    var transferProgress by remember { mutableStateOf<Float?>(null) }

    /** Register only after the file exists and has passed the LiteRT-LM magic check. */
    suspend fun registerModel(file: File) {
        prefs.addInstalledModel(LocalRuntime.LiteRT, file.name, file.absolutePath)
        val metadata = LiteRtModelMetadata.deriveCapabilities(file.name)
        val config = LiteRtModelDefaults.forModelFile(file.name)
        val ceiling = LiteRtModelDefaults.contextCeilingFromFileName(file.name)
            ?: config.maxContextLength
            ?: config.maxTokens
        val newModel = Model(
            modelId = file.name,
            displayName = LiteRtCatalog.displayNameFor(file.name),
            inputModalities = metadata.inputModalities,
            abilities = metadata.abilities,
            contextLength = ceiling,
        )
        val active = currentProvider
        if (active.models.none { it.modelId == file.name }) {
            currentOnEdit(active.copy(models = active.models + newModel))
        }
        // This tab normally saves only via its floating Save button. Model imports are
        // irreversible disk operations, so persist the model registration immediately.
        settingsStore.update { settings ->
            settings.copy(providers = settings.providers.map { setting ->
                if (setting is ProviderSetting.LiteRtLocal && setting.id == active.id &&
                    setting.models.none { it.modelId == file.name }
                ) setting.copy(models = setting.models + newModel) else setting
            })
        }
    }

    fun submit(task: suspend () -> Unit) {
        if (busy) return
        busy = true
        status = null
        scope.launch {
            try {
                task()
            } catch (e: Exception) {
                status = e.message ?: e.javaClass.simpleName
            } finally {
                busy = false
                transferProgress = null
            }
        }
    }

    val importPicker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        submit {
            val file = withContext(Dispatchers.IO) { importLiteRtFile(context, uri) }
            registerModel(file)
            status = "已导入 ${file.name}"
        }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Text("LiteRT-LM 本地模型 (.litertlm)")
            Text("模型文件保存在应用私有目录；本地推理无需 API Key。首次加载可能较慢。")
            Button(onClick = { importPicker.launch("*/*") }, enabled = !busy) {
                Text("从文件导入 .litertlm")
            }
            OutlinedTextField(
                value = url,
                onValueChange = { url = it },
                label = { Text("模型直链（可选，支持 Hugging Face resolve/blob URL）") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                enabled = !busy,
            )
            OutlinedButton(
                onClick = {
                    submit {
                        val normalized = ModelInstall.normalizeHuggingFaceUrl(url.trim())
                        require(ModelInstall.isValidDownloadUrl(normalized)) { "请提供 HTTPS 模型下载地址" }
                        val name = safeModelFileName(ModelInstall.extractFileNameFromUrl(normalized))
                        val target = ModelInstall.targetFile(ModelInstall.localModelsDir(context), LocalRuntime.LiteRT, name)
                        require(!target.exists()) { "模型文件已存在：$name" }
                        var completed: File? = null
                        withContext(Dispatchers.IO) {
                            ModelInstall.download(client, normalized, target).collect { progress ->
                                when (progress) {
                                    is ModelInstall.Progress.Started -> transferProgress = null
                                    is ModelInstall.Progress.Tick -> {
                                        transferProgress = progress.totalBytes?.takeIf { it > 0 }
                                            ?.let { (progress.bytesRead.toDouble() / it).toFloat().coerceIn(0f, 1f) }
                                    }
                                    is ModelInstall.Progress.Done -> completed = progress.file
                                    is ModelInstall.Progress.Failed -> throw progress.cause
                                }
                            }
                        }
                        val file = completed ?: error("下载未完成")
                        registerModel(file)
                        status = "已安装 ${file.name}"
                    }
                },
                enabled = !busy && url.isNotBlank(),
            ) { Text("从 URL 安装") }
            if (busy) {
                val progress = transferProgress
                if (progress == null) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                else LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
            }
            status?.let { Text(it) }
        }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("运行时")
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("强制 CPU（关闭后可尝试 GPU）")
                Switch(
                    checked = forceCpu,
                    onCheckedChange = { value -> submit { prefs.setForceCpu(LocalRuntime.LiteRT, value) } },
                    enabled = !busy,
                )
            }
            Text("Tensor / GrapheneOS 设备建议先使用 CPU；GPU 原生崩溃不能通过普通异常捕获。")
            OutlinedTextField(
                value = tokenField,
                onValueChange = { tokenField = it.filter(Char::isDigit) },
                label = { Text("maxNumTokens（留空使用模型默认值）") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                enabled = !busy,
            )
            OutlinedButton(
                onClick = {
                    submit {
                        val value = tokenField.takeIf(String::isNotBlank)?.toIntOrNull()
                        require(tokenField.isBlank() || (value != null && value in 128..32768)) {
                            "请输入 128–32768，或留空"
                        }
                        prefs.setMaxNumTokensOverride(LocalRuntime.LiteRT, value)
                        runtime.closeIfLoaded() // New EngineConfig takes effect on next request.
                        status = "上下文设置已保存；实际值仍受模型 KV Cache 上限约束"
                    }
                },
                enabled = !busy,
            ) { Text("保存上下文设置") }
            Text("实际运行的上下文不会超过模型文件自身的缓存上限。")
        }
    }

    Card(modifier = Modifier.fillMaxWidth()) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("已安装模型")
            if (provider.models.isEmpty()) Text("尚未导入模型")
            provider.models.forEach { model ->
                HorizontalDivider()
                Text(model.displayName.ifBlank { model.modelId })
                Text(model.modelId)
                if (model.modelId !in installed) Text("文件未登记；请重新导入模型")
                TextButton(
                    enabled = !busy,
                    onClick = {
                        submit {
                            runtime.closeIfLoaded()
                            val path = installed[model.modelId]
                            prefs.removeInstalledModel(LocalRuntime.LiteRT, model.modelId)
                            if (path != null) {
                                withContext(Dispatchers.IO) {
                                    val file = File(path)
                                    val ownedRoot = ModelInstall.localModelsDir(context).canonicalFile
                                    if (file.canonicalFile.toPath().startsWith(ownedRoot.toPath())) file.delete()
                                }
                            }
                            val providerId = currentProvider.id
                            currentOnEdit(currentProvider.copy(models = currentProvider.models.filterNot { it.id == model.id }))
                            settingsStore.update { settings ->
                                settings.copy(providers = settings.providers.map { setting ->
                                    if (setting is ProviderSetting.LiteRtLocal && setting.id == providerId) {
                                        setting.copy(models = setting.models.filterNot { it.modelId == model.modelId })
                                    } else setting
                                })
                            }
                            status = "已移除 ${model.modelId}"
                        }
                    },
                ) { Text("移除模型") }
            }
        }
    }
}

private fun safeModelFileName(raw: String): String {
    val name = raw.substringAfterLast('/').substringAfterLast('\\').trim()
    require(name.lowercase().endsWith(".litertlm") && name != ".litertlm") {
        "请选择 .litertlm 模型文件"
    }
    require(name.none { it == '/' || it == '\\' || it == '\u0000' }) { "文件名不安全" }
    return name
}

/** Stream-copy large SAF files; never materialize model weights as a ByteArray. */
private fun importLiteRtFile(context: Context, uri: Uri): File {
    val rawName = context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
        ?.use { cursor ->
            if (cursor.moveToFirst()) cursor.getString(0) else null
        } ?: uri.lastPathSegment.orEmpty()
    val fileName = safeModelFileName(rawName)
    val target = ModelInstall.targetFile(ModelInstall.localModelsDir(context), LocalRuntime.LiteRT, fileName)
    require(!target.exists()) { "模型文件已存在：$fileName" }
    target.parentFile?.mkdirs()
    val partial = File(target.path + ".importing")
    try {
        context.contentResolver.openInputStream(uri)?.use { input ->
            partial.outputStream().use { output -> input.copyTo(output) }
        } ?: error("无法读取选择的文件")
        require(partial.inputStream().use { input ->
            val header = ByteArray(16)
            val count = input.read(header)
            ModelInstall.isValidMagicForExtension("litertlm", if (count > 0) header.copyOf(count) else byteArrayOf())
        }) { "文件不是有效的 LiteRT-LM 模型" }
        require(partial.renameTo(target)) { "无法保存模型文件" }
        return target
    } finally {
        partial.delete()
    }
}
