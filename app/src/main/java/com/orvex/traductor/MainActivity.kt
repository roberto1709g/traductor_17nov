package com.orvex.traductor

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.orvex.traductor.audio.SpanishVoiceOption
import com.orvex.traductor.audio.TtsBluetoothManager
import com.orvex.traductor.translation.TranslationManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.net.URL
import java.util.concurrent.atomic.AtomicReference
import kotlin.coroutines.resume
import kotlin.math.sqrt

class MainActivity : ComponentActivity() {

    init {
        System.loadLibrary("whisper_jni")
    }

    private external fun transcribeAudioNative(modelPath: String, audioData: FloatArray): String
    private external fun freeWhisperNative()

    private val translationManager = TranslationManager()
    private var ttsManager: TtsBluetoothManager? = null
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Main)

    private var captureJob: Job? = null
    private var processorJob: Job? = null

    // Cola de 1 solo bloque (conflated) para evitar acumulación si Whisper tarda más
    private val latestAudioChunk = AtomicReference<FloatArray?>(null)

    private val requestPermissionsLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { _ ->
        ttsManager?.updateAudioRouteStatus()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Mantener pantalla encendida mientras interpreta una película
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        requestRequiredPermissions()

        setContent {
            var ttsStatus by remember { mutableStateOf("Inicializando TTS...") }
            var btStatus by remember { mutableStateOf("Comprobando salida de audio...") }
            var isSpeaking by remember { mutableStateOf(false) }
            var availableVoices by remember { mutableStateOf<List<SpanishVoiceOption>>(emptyList()) }
            var selectedVoiceId by remember { mutableStateOf("") }

            DisposableEffect(Unit) {
                val manager = TtsBluetoothManager(
                    context = this@MainActivity,
                    onTtsStatusChanged = { ttsStatus = it },
                    onSpeakingStateChanged = { isSpeaking = it },
                    onBluetoothStatusChanged = { btStatus = it },
                    onVoicesLoaded = { voices ->
                        availableVoices = voices
                        voices.firstOrNull { it.isOffline }?.let {
                            selectedVoiceId = it.id
                        }
                    }
                )
                ttsManager = manager
                onDispose {
                    stopContinuousInterpreter()
                    manager.release()
                    ttsManager = null
                }
            }

            Surface(
                modifier = Modifier.fillMaxSize(),
                color = MaterialTheme.colorScheme.background
            ) {
                MainScreen(
                    ttsStatus = ttsStatus,
                    btStatus = btStatus,
                    isSpeaking = isSpeaking,
                    availableVoices = availableVoices,
                    selectedVoiceId = selectedVoiceId,
                    onSelectVoice = { option ->
                        selectedVoiceId = option.id
                        ttsManager?.selectVoice(option)
                    },
                    onSpeechRateChange = { rate ->
                        ttsManager?.setSpeechRate(rate)
                    },
                    onMuteToggle = { muted ->
                        ttsManager?.setMuted(muted)
                    },
                    onStopSpeech = {
                        ttsManager?.stopSpeaking()
                    },
                    onTestSpeech = {
                        ttsManager?.speak("Modo intérprete de películas listo en tus auriculares.")
                    },
                    onDownloadWhisper = { onStatus -> downloadWhisperModel(onStatus) },
                    onDownloadTranslation = { onStatus ->
                        translationManager.downloadModel { _, msg ->
                            onStatus(msg)
                        }
                    },
                    onStartContinuous = { rmsThreshold, onProgress, onNewSegment, onRmsUpdate ->
                        startContinuousInterpreter(rmsThreshold, onProgress, onNewSegment, onRmsUpdate)
                    },
                    onStopContinuous = {
                        stopContinuousInterpreter()
                    }
                )
            }
        }
    }

    private fun requestRequiredPermissions() {
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            permissions.add(Manifest.permission.BLUETOOTH_CONNECT)
        }
        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isNotEmpty()) {
            requestPermissionsLauncher.launch(missing.toTypedArray())
        }
    }

    private fun downloadWhisperModel(onStatus: (String) -> Unit) {
        serviceScope.launch(Dispatchers.IO) {
            try {
                withContext(Dispatchers.Main) { onStatus("Descargando modelo Whisper...") }
                val modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin"
                val file = File(filesDir, "ggml-tiny.en.bin")

                if (file.exists() && file.length() > 1024 * 1024) {
                    withContext(Dispatchers.Main) {
                        onStatus("Modelo Whisper listo (${file.length() / (1024 * 1024)} MB).")
                    }
                    return@launch
                }

                URL(modelUrl).openStream().use { input ->
                    FileOutputStream(file).use { output ->
                        input.copyTo(output)
                    }
                }
                withContext(Dispatchers.Main) { onStatus("Modelo Whisper descargado con éxito.") }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) { onStatus("Error al descargar Whisper: ${e.message}") }
            }
        }
    }

    private fun startContinuousInterpreter(
        rmsThreshold: Float,
        onStatusUpdate: (String) -> Unit,
        onSegmentTranslated: (String, String, Long) -> Unit,
        onRmsLevel: (Float) -> Unit
    ) {
        stopContinuousInterpreter()

        val modelFile = File(filesDir, "ggml-tiny.en.bin")
        if (!modelFile.exists() || modelFile.length() < 1024 * 1024) {
            onStatusUpdate("Error: Descarga primero el modelo Whisper.")
            return
        }

        if (ActivityCompat.checkSelfPermission(applicationContext, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            onStatusUpdate("Error: Permiso de micrófono denegado.")
            return
        }

        latestAudioChunk.set(null)

        // 1. PRODUCTOR: Captura continua con ventana deslizante (4.0s ventana, 0.7s solapamiento)
        captureJob = serviceScope.launch(Dispatchers.IO) {
            val sampleRate = 16000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBuffer = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

            val audioRecord = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                sampleRate,
                channelConfig,
                audioFormat,
                maxOf(minBuffer, sampleRate * 2)
            )

            if (audioRecord.state != AudioRecord.STATE_INITIALIZED) {
                withContext(Dispatchers.Main) {
                    onStatusUpdate("Error al abrir el micrófono.")
                }
                return@launch
            }

            val windowSamples = sampleRate * 4          // 4.0 segundos (64,000 muestras)
            val overlapSamples = (sampleRate * 0.7).toInt() // 0.7 segundos de solapamiento (11,200 muestras)
            val stepSamples = windowSamples - overlapSamples // 3.3 segundos nuevos por ciclo

            val ringBuffer = FloatArray(windowSamples)
            var ringFill = 0
            val readShortBuffer = ShortArray(1600) // bloques de 100 ms para medidor RMS en vivo

            try {
                audioRecord.startRecording()
                withContext(Dispatchers.Main) {
                    onStatusUpdate("Escuchando película continuamente...")
                }

                while (isActive) {
                    val read = audioRecord.read(readShortBuffer, 0, readShortBuffer.size)
                    if (read > 0) {
                        var sumSquares = 0.0
                        for (i in 0 until read) {
                            val sample = readShortBuffer[i] / 32768.0f
                            sumSquares += sample * sample
                            if (ringFill < windowSamples) {
                                ringBuffer[ringFill++] = sample
                            }
                        }

                        // Actualizar nivel RMS visual cada 100 ms
                        val blockRms = sqrt(sumSquares / read).toFloat()
                        withContext(Dispatchers.Main) {
                            onRmsLevel(blockRms)
                        }

                        // Cuando completamos la ventana de 4.0 segundos
                        if (ringFill >= windowSamples) {
                            val windowCopy = ringBuffer.copyOf()

                            // Desplazar los últimos 0.7s al inicio para evitar cortar palabras
                            System.arraycopy(ringBuffer, stepSamples, ringBuffer, 0, overlapSamples)
                            ringFill = overlapSamples

                            // Evaluar energía RMS global de la ventana para filtrar silencios
                            var totalSq = 0.0
                            for (v in windowCopy) {
                                totalSq += v * v
                            }
                            val windowRms = sqrt(totalSq / windowCopy.size).toFloat()

                            if (windowRms >= rmsThreshold) {
                                // Reemplazar el bloque pendiente (si Whisper aún estaba ocupado, no acumula cola)
                                latestAudioChunk.set(windowCopy)
                            } else {
                                withContext(Dispatchers.Main) {
                                    onStatusUpdate("Silencio detectado (RMS: ${"%.3f".format(windowRms)}) — esperando diálogo...")
                                }
                            }
                        }
                    } else {
                        delay(10)
                    }
                }
            } finally {
                try {
                    audioRecord.stop()
                } catch (_: Exception) {}
                audioRecord.release()
                withContext(Dispatchers.Main) {
                    onRmsLevel(0f)
                }
            }
        }

        // 2. CONSUMIDOR: Inferencia Whisper -> Filtro anti-repetición -> ML Kit -> Voz TTS BT
        processorJob = serviceScope.launch(Dispatchers.IO) {
            var lastEnglishPhrase = ""
            var lastSpanishPhrase = ""

            while (isActive) {
                val chunk = latestAudioChunk.getAndSet(null)
                if (chunk == null) {
                    delay(80)
                    continue
                }

                withContext(Dispatchers.Main) {
                    onStatusUpdate("Transcribiendo fragmento con Whisper...")
                }

                val t0 = SystemClock.elapsedRealtime()
                val rawEnglish = transcribeAudioNative(modelFile.absolutePath, chunk).trim()
                val latencyMs = SystemClock.elapsedRealtime() - t0

                // Filtrar ruido alucinado típico de Whisper en pausas musicales
                val cleanedEnglish = cleanWhisperArtifacts(rawEnglish)
                if (cleanedEnglish.isBlank()) {
                    withContext(Dispatchers.Main) {
                        onStatusUpdate("Sin diálogo claro (${latencyMs} ms) — escuchando...")
                    }
                    continue
                }

                // Eliminar solapamiento textual con la frase inmediatamente anterior
                val deduplicatedEnglish = removeOverlapDuplicate(lastEnglishPhrase, cleanedEnglish)
                if (deduplicatedEnglish.isBlank() || deduplicatedEnglish.equals(lastEnglishPhrase, ignoreCase = true)) {
                    withContext(Dispatchers.Main) {
                        onStatusUpdate("Fragmento repetido omitido (${latencyMs} ms)")
                    }
                    continue
                }
                lastEnglishPhrase = cleanedEnglish

                withContext(Dispatchers.Main) {
                    onStatusUpdate("Traduciendo (${latencyMs} ms)...")
                }

                val spanish = translateSuspend(deduplicatedEnglish).trim()
                if (spanish.isNotBlank() && !spanish.equals(lastSpanishPhrase, ignoreCase = true)) {
                    lastSpanishPhrase = spanish
                    withContext(Dispatchers.Main) {
                        onSegmentTranslated(deduplicatedEnglish, spanish, latencyMs)
                        onStatusUpdate("Activo · Último ciclo: ${latencyMs} ms")
                        ttsManager?.speak(spanish)
                    }
                }
            }
        }
    }

    private suspend fun translateSuspend(text: String): String =
        suspendCancellableCoroutine { cont ->
            translationManager.translateText(text) { result ->
                if (cont.isActive) {
                    cont.resume(result)
                }
            }
        }

    private fun cleanWhisperArtifacts(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isBlank()) return ""
        val lower = trimmed.lowercase()
        // Descartar etiquetas no verbales de Whisper en películas
        if (lower.startsWith("[") && lower.endsWith("]")) return ""
        if (lower.startsWith("(") && lower.endsWith(")")) return ""
        if (lower == "you" || lower == "you." || lower == "thank you." || lower == "thanks for watching.") {
            // Alucinaciones clásicas de Whisper tiny.en cuando hay música de fondo
            return ""
        }
        return trimmed
    }

    private fun removeOverlapDuplicate(previous: String, current: String): String {
        if (previous.isBlank()) return current
        val prevWords = previous.lowercase().replace(Regex("[^a-z0-9\s']"), "").split(Regex("\s+")).filter { it.isNotBlank() }
        val currRawWords = current.split(Regex("\s+")).filter { it.isNotBlank() }
        val currNormWords = currRawWords.map { it.lowercase().replace(Regex("[^a-z0-9']"), "") }

        if (prevWords.isEmpty() || currNormWords.isEmpty()) return current

        // Buscar si las primeras 1..4 palabras del fragmento actual coinciden con el final del anterior
        val maxCheck = minOf(4, prevWords.size, currNormWords.size)
        for (k in maxCheck downTo 1) {
            val prevTail = prevWords.takeLast(k)
            val currHead = currNormWords.take(k)
            if (prevTail == currHead) {
                return currRawWords.drop(k).joinToString(" ")
            }
        }
        return current
    }

    private fun stopContinuousInterpreter() {
        captureJob?.cancel()
        processorJob?.cancel()
        captureJob = null
        processorJob = null
        latestAudioChunk.set(null)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopContinuousInterpreter()
        freeWhisperNative()
        translationManager.close()
    }
}

@Composable
fun MainScreen(
    ttsStatus: String,
    btStatus: String,
    isSpeaking: Boolean,
    availableVoices: List<SpanishVoiceOption>,
    selectedVoiceId: String,
    onSelectVoice: (SpanishVoiceOption) -> Unit,
    onSpeechRateChange: (Float) -> Unit,
    onMuteToggle: (Boolean) -> Unit,
    onStopSpeech: () -> Unit,
    onTestSpeech: () -> Unit,
    onDownloadWhisper: ((String) -> Unit) -> Unit,
    onDownloadTranslation: ((String) -> Unit) -> Unit,
    onStartContinuous: (Float, (String) -> Unit, (String, String, Long) -> Unit, (Float) -> Unit) -> Unit,
    onStopContinuous: () -> Unit
) {
    var whisperStatus by remember { mutableStateOf("Comprobar estado") }
    var translationStatus by remember { mutableStateOf("Pendiente") }
    var recognizedHistory by remember { mutableStateOf("Aquí aparecerá el diálogo original en inglés...") }
    var translatedHistory by remember { mutableStateOf("Aquí aparecerá la traducción en español...") }
    var processingState by remember { mutableStateOf("Listo para iniciar modo continuo") }
    var lastLatencyMs by remember { mutableStateOf(0L) }
    var isRunning by remember { mutableStateOf(false) }
    var isVoiceEnabled by remember { mutableStateOf(true) }
    var selectedRate by remember { mutableStateOf(1.25f) }
    var rmsThreshold by remember { mutableStateOf(0.012f) }
    var currentRms by remember { mutableStateOf(0f) }
    var voiceMenuExpanded by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text(text = "TRADUCTOR V0.5 · INTÉRPRETE CONTINUO", style = MaterialTheme.typography.titleLarge)

        // SALIDA BLUETOOTH Y VOZ TTS
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(text = "AUDIO BLUETOOTH Y VOZ ESPAÑOLA", style = MaterialTheme.typography.titleMedium)
                Text(text = btStatus, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary)
                Text(text = ttsStatus, style = MaterialTheme.typography.bodySmall)

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = "Voz automática en auriculares", style = MaterialTheme.typography.bodyMedium)
                    Switch(
                        checked = isVoiceEnabled,
                        onCheckedChange = { enabled ->
                            isVoiceEnabled = enabled
                            onMuteToggle(!enabled)
                        }
                    )
                }

                Text(text = "Velocidad de habla: ${selectedRate}x", style = MaterialTheme.typography.labelMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(1.0f, 1.15f, 1.25f, 1.45f).forEach { rate ->
                        FilterChip(
                            selected = selectedRate == rate,
                            onClick = {
                                selectedRate = rate
                                onSpeechRateChange(rate)
                            },
                            label = { Text("${rate}x") }
                        )
                    }
                }

                if (availableVoices.isNotEmpty()) {
                    Box {
                        OutlinedButton(
                            modifier = Modifier.fillMaxWidth(),
                            onClick = { voiceMenuExpanded = true }
                        ) {
                            val currentLabel = availableVoices.find { it.id == selectedVoiceId }?.label ?: "Seleccionar voz española"
                            Text(currentLabel)
                        }
                        DropdownMenu(
                            expanded = voiceMenuExpanded,
                            onDismissRequest = { voiceMenuExpanded = false }
                        ) {
                            availableVoices.forEach { option ->
                                DropdownMenuItem(
                                    text = { Text(option.label) },
                                    onClick = {
                                        onSelectVoice(option)
                                        voiceMenuExpanded = false
                                    }
                                )
                            }
                        }
                    }
                }

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(modifier = Modifier.weight(1f), onClick = onTestSpeech) {
                        Text("PROBAR VOZ BT")
                    }
                    OutlinedButton(modifier = Modifier.weight(1f), enabled = isSpeaking, onClick = onStopSpeech) {
                        Text("CALLAR VOZ")
                    }
                }
            }
        }

        // CALIBRACIÓN DE SILENCIO Y MODELOS
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(text = "SENSIBILIDAD DE MICRÓFONO (FILTRO SILENCIO)", style = MaterialTheme.typography.titleSmall)
                Text(
                    text = "Umbral RMS: ${"%.3f".format(rmsThreshold)} | Nivel actual: ${"%.3f".format(currentRms)}",
                    style = MaterialTheme.typography.bodySmall
                )
                LinearProgressIndicator(
                    progress = { (currentRms * 10f).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth()
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(
                        0.005f to "Alta (TV baja)",
                        0.012f to "Normal",
                        0.025f to "Baja (Ruido)"
                    ).forEach { (thresh, label) ->
                        FilterChip(
                            selected = rmsThreshold == thresh,
                            onClick = { rmsThreshold = thresh },
                            label = { Text(label) }
                        )
                    }
                }

                HorizontalDivider()

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        modifier = Modifier.weight(1f),
                        enabled = !isRunning,
                        onClick = { onDownloadWhisper { whisperStatus = it } }
                    ) {
                        Text("WHISPER")
                    }
                    Button(
                        modifier = Modifier.weight(1f),
                        enabled = !isRunning,
                        onClick = { onDownloadTranslation { translationStatus = it } }
                    ) {
                        Text("TRADUCCIÓN")
                    }
                }
                Text(text = "Whisper: $whisperStatus | Traducción: $translationStatus", style = MaterialTheme.typography.bodySmall)
            }
        }

        // BOTÓN PRINCIPAL DE MODO INTÉRPRETE CONTINUO
        Button(
            modifier = Modifier.fillMaxWidth(),
            colors = if (isRunning) {
                ButtonDefaults.buttonColors(containerColor = MaterialTheme.colorScheme.error)
            } else {
                ButtonDefaults.buttonColors()
            },
            onClick = {
                if (!isRunning) {
                    isRunning = true
                    recognizedHistory = ""
                    translatedHistory = ""
                    onStartContinuous(
                        rmsThreshold,
                        { status -> processingState = status },
                        { eng, esp, lat ->
                            lastLatencyMs = lat
                            recognizedHistory = if (recognizedHistory.isBlank()) eng else "$eng
• $recognizedHistory".take(600)
                            translatedHistory = if (translatedHistory.isBlank()) esp else "$esp
• $translatedHistory".take(600)
                        },
                        { rms -> currentRms = rms }
                    )
                } else {
                    isRunning = false
                    onStopContinuous()
                    processingState = "Intérprete detenido."
                }
            }
        ) {
            Text(if (isRunning) "DETENER INTÉRPRETE CONTINUO" else "INICIAR INTÉRPRETE DE PELÍCULA (CONTINUO)")
        }

        Text(
            text = if (lastLatencyMs > 0) "$processingState | Inferencia: ${lastLatencyMs} ms" else processingState,
            style = MaterialTheme.typography.bodyMedium
        )

        OutlinedCard(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(text = "DIÁLOGO ORIGINAL (INGLÉS)", style = MaterialTheme.typography.labelMedium)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = recognizedHistory.ifBlank { "Esperando diálogo en inglés..." },
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }

        OutlinedCard(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text(text = "TRADUCCIÓN EN VIVO (ESPAÑOL)", style = MaterialTheme.typography.labelMedium)
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = translatedHistory.ifBlank { "Esperando traducción..." },
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}
