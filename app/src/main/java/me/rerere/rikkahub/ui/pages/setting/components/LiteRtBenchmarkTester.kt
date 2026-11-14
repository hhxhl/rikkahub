package me.rerere.rikkahub.ui.pages.setting.components

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import me.rerere.ai.provider.ModelType
import me.rerere.ai.provider.ProviderSetting
import me.rerere.hugeicons.HugeIcons
import me.rerere.hugeicons.stroke.Tools
import me.rerere.locallm.LocalRuntimePreferences
import me.rerere.locallm.litert.LiteRtBenchmarkRunner
import me.rerere.locallm.litert.LiteRtRuntime
import me.rerere.rikkahub.BuildConfig
import me.rerere.rikkahub.R
import me.rerere.rikkahub.ui.components.ai.ModelSelector
import org.koin.compose.koinInject
import java.util.Locale

/** Only shown for ProviderSetting.LiteRtLocal; original connection testing is unchanged. */
@Composable
fun LiteRtBenchmarkTester(provider: ProviderSetting.LiteRtLocal) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val runtime = koinInject<LiteRtRuntime>()
    val preferences = koinInject<LocalRuntimePreferences>()
    val runner = remember(context, runtime, preferences) {
        LiteRtBenchmarkRunner(context.applicationContext, runtime, preferences)
    }

    var dialogOpen by remember { mutableStateOf(false) }
    var model by remember(provider.id) {
        mutableStateOf(provider.models.firstOrNull { it.type == ModelType.CHAT })
    }
    var accelerator by remember { mutableStateOf("CPU") }
    var prefill by remember { mutableStateOf("256") }
    var decode by remember { mutableStateOf("256") }
    var runCount by remember { mutableStateOf("3") }
    var running by remember { mutableStateOf(false) }
    var completed by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }
    var result by remember { mutableStateOf<LiteRtBenchmarkRunner.Result?>(null) }
    var exported by remember { mutableStateOf(false) }

    val exportLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/csv"),
    ) { uri ->
        val snapshot = result
        if (uri != null && snapshot != null) {
            scope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        checkNotNull(context.contentResolver.openOutputStream(uri))
                            .bufferedWriter(Charsets.UTF_8).use {
                                it.write(snapshot.toCsv(BuildConfig.VERSION_NAME))
                            }
                    }
                    exported = true
                } catch (e: Exception) {
                    error = e.message ?: e.javaClass.simpleName
                }
            }
        }
    }

    IconButton(onClick = { dialogOpen = true }) {
        Icon(HugeIcons.Tools, contentDescription = stringResource(R.string.litert_benchmark_title))
    }

    if (dialogOpen) {
        val p = prefill.toIntOrNull()
        val d = decode.toIntOrNull()
        val n = runCount.toIntOrNull()
        val valid = model != null && p != null && p in 1..8192 &&
            d != null && d in 1..4096 && n != null && n in 1..10
        AlertDialog(
            onDismissRequest = { if (!running) dialogOpen = false },
            title = { Text(stringResource(R.string.litert_benchmark_title)) },
            text = {
                Column(
                    modifier = Modifier.heightIn(max = 520.dp).verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    Text(stringResource(R.string.litert_benchmark_tip))
                    ModelSelector(
                        modelId = model?.id,
                        providers = listOf(provider),
                        type = ModelType.CHAT,
                        modifier = Modifier.fillMaxWidth(),
                    ) { if (!running) model = it }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf("CPU", "GPU").forEach { backend ->
                            FilterChip(
                                selected = accelerator == backend,
                                onClick = { if (!running) accelerator = backend },
                                label = { Text(backend) },
                                enabled = !running,
                            )
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedTextField(
                            value = prefill, onValueChange = { prefill = it },
                            label = { Text(stringResource(R.string.litert_benchmark_prefill)) },
                            enabled = !running, singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.weight(1f),
                        )
                        OutlinedTextField(
                            value = decode, onValueChange = { decode = it },
                            label = { Text(stringResource(R.string.litert_benchmark_decode)) },
                            enabled = !running, singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.weight(1f),
                        )
                    }
                    OutlinedTextField(
                        value = runCount, onValueChange = { runCount = it },
                        label = { Text(stringResource(R.string.litert_benchmark_runs)) },
                        enabled = !running, singleLine = true,
                        keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                        modifier = Modifier.fillMaxWidth(),
                    )
                    if (running) {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                        Text(stringResource(R.string.litert_benchmark_progress, completed, n ?: 0))
                    }
                    error?.let { Text(stringResource(R.string.litert_benchmark_error, it)) }
                    result?.let { stats ->
                        Text(stringResource(R.string.litert_benchmark_results))
                        Text("${stats.accelerator} · ${stats.modelName}")
                        Text(stringResource(R.string.litert_benchmark_avg,
                            formatMetric(stats.prefillTps), formatMetric(stats.decodeTps),
                            formatMetric(stats.ttftSeconds)))
                        Text(stringResource(R.string.litert_benchmark_init,
                            formatMetric(stats.firstInitMs),
                            stats.steadyInitMs?.let(::formatMetric) ?: "—"))
                        stats.runs.forEach { run ->
                            Text(stringResource(R.string.litert_benchmark_one_run,
                                run.index, formatMetric(run.metrics.lastPrefillTokensPerSecond),
                                formatMetric(run.metrics.lastDecodeTokensPerSecond),
                                formatMetric(run.metrics.timeToFirstTokenInSecond)))
                        }
                        if (exported) Text(stringResource(R.string.litert_benchmark_exported))
                    }
                }
            },
            dismissButton = {
                TextButton(onClick = { dialogOpen = false }, enabled = !running) {
                    Text(stringResource(R.string.cancel))
                }
            },
            confirmButton = {
                Row {
                    if (!running && result != null) {
                        TextButton(onClick = {
                            exported = false
                            exportLauncher.launch("litert-benchmark-${System.currentTimeMillis()}.csv")
                        }) { Text(stringResource(R.string.litert_benchmark_export)) }
                    }
                    TextButton(
                        enabled = !running && valid,
                        onClick = {
                            val id = model?.modelId ?: return@TextButton
                            val prefillCount = p ?: return@TextButton
                            val decodeCount = d ?: return@TextButton
                            val repeatCount = n ?: return@TextButton
                            val selectedAccelerator = accelerator
                            running = true
                            completed = 0
                            error = null
                            result = null
                            exported = false
                            scope.launch {
                                try {
                                    result = runner.run(
                                        modelId = id,
                                        accelerator = selectedAccelerator,
                                        prefillTokens = prefillCount,
                                        decodeTokens = decodeCount,
                                        runCount = repeatCount,
                                        onProgress = { count ->
                                            withContext(Dispatchers.Main) { completed = count }
                                        },
                                    )
                                } catch (e: CancellationException) {
                                    throw e
                                } catch (e: Exception) {
                                    error = e.message ?: e.javaClass.simpleName
                                } finally {
                                    running = false
                                }
                            }
                        },
                    ) { Text(stringResource(R.string.litert_benchmark_start)) }
                }
            },
        )
    }
}

private fun formatMetric(value: Double): String =
    if (value.isFinite()) String.format(Locale.US, "%.2f", value) else "—"
