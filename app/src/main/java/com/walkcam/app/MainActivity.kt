package com.walkcam.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: SegOverlayView
    private lateinit var hudText: TextView
    private lateinit var btnOutdoor: Button
    private lateinit var btnIndoor: Button

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    @Volatile private var engine: SegEngine? = null
    private var lastProcessAt = 0L
    private val rgb = IntArray(YuvToRgb.OUT * YuvToRgb.OUT)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            start()
        } else {
            Toast.makeText(this, "需要相机权限才能使用", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        hudText = findViewById(R.id.hudText)
        btnOutdoor = findViewById(R.id.btnOutdoor)
        btnIndoor = findViewById(R.id.btnIndoor)

        btnOutdoor.setOnClickListener { switchMode(0) }
        btnIndoor.setOnClickListener { switchMode(1) }
        refreshModeButtons(0)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            start()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun switchMode(m: Int) {
        val e = engine
        if (e == null) {
            Toast.makeText(this, "模型还在加载中", Toast.LENGTH_SHORT).show()
            return
        }
        e.setMode(m)
        refreshModeButtons(m)
        overlayView.clear()
        Toast.makeText(this, "已切换到${e.modeNames[m]}模式：${e.labelsFor(m)}", Toast.LENGTH_LONG).show()
    }

    private fun refreshModeButtons(m: Int) {
        btnOutdoor.isActivated = m == 0
        btnIndoor.isActivated = m == 1
        btnOutdoor.alpha = if (m == 0) 1f else 0.5f
        btnIndoor.alpha = if (m == 1) 1f else 0.5f
    }

    private fun start() {
        hudText.text = "模型加载中（室外+室内），请稍候…"
        Thread {
            try {
                val e = SegEngine(this)
                e.warmup()
                engine = e
                runOnUiThread {
                    hudText.text = "就绪（室外模式）"
                    startCamera()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "init failed", t)
                runOnUiThread { hudText.text = "初始化失败：${t.javaClass.simpleName}: ${t.message}" }
            }
        }.start()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(cameraExecutor, ::onFrame)
                provider.unbindAll()
                provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
            } catch (t: Throwable) {
                Log.e(TAG, "camera failed", t)
                hudText.text = "相机启动失败：${t.message}"
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onFrame(image: ImageProxy) {
        val e = engine
        if (e == null) {
            image.close()
            return
        }
        val nowMs = System.currentTimeMillis()
        if (nowMs - lastProcessAt < THROTTLE_MS || !busy.compareAndSet(false, true)) {
            image.close()
            return
        }
        lastProcessAt = nowMs
        try {
            val t0 = System.currentTimeMillis()
            val info: YuvToRgb.FrameInfo
            try {
                info = YuvToRgb.convert(image, rgb)
            } finally {
                image.close()
            }
            val res = e.run(rgb)
            val totalMs = System.currentTimeMillis() - t0
            runOnUiThread {
                hudText.text = String.format(
                    Locale.CHINA,
                    "模式 %s | 分割 %d ms | 端到端 %d ms\n画面中央 %.0f%% 可通行",
                    e.modeNames[e.mode], res.ms, totalMs, res.walkPct
                )
                overlayView.update(res.walkable, res.maskSize, info)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "frame error", t)
        } finally {
            busy.set(false)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        engine?.close()
    }

    companion object {
        private const val TAG = "WalkCam"
        private const val THROTTLE_MS = 1000L
    }
}
