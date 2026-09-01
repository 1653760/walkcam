package com.walkcam.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.FloatBuffer

class SegEngine(context: Context) {

    data class SegResult(val walkable: ByteArray, val maskSize: Int, val walkPct: Float, val ms: Long)

    private val env = OrtEnvironment.getEnvironment()
    private val sessions = ArrayList<OrtSession>(2)
    private val inputNames = ArrayList<String>(2)
    private val walkSets = ArrayList<Set<Int>>(2)
    private val walkLabels = ArrayList<String>(2)
    val modeNames = arrayOf("室外", "室内")

    @Volatile var mode = 0

    private val inputSize = 512
    private val maskSize = 128
    private val numClasses = intArrayOf(19, 150)
    private val floatBuf = FloatArray(3 * inputSize * inputSize)
    private val means = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val stds = floatArrayOf(0.229f, 0.224f, 0.225f)

    init {
        val spec = context.assets.open("walkable.json").bufferedReader().use { it.readText() }
        val root = JSONObject(spec)
        loadModel(context, "seg_outdoor.onnx", root.getJSONObject("outdoor"))
        loadModel(context, "seg_indoor.onnx", root.getJSONObject("indoor"))
        Log.i(TAG, "seg engine ready: outdoor classes=${numClasses[0]}, indoor classes=${numClasses[1]}")
    }

    private fun loadModel(context: Context, asset: String, spec: JSONObject) {
        val bytes = context.assets.open(asset).readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(bytes, opts)
        sessions.add(session)
        inputNames.add(session.inputInfo.keys.first())
        val walkIds = spec.getJSONArray("walkable").let { arr ->
            val s = HashSet<Int>(arr.length())
            for (i in 0 until arr.length()) s.add(arr.getInt(i))
            s
        }
        walkSets.add(walkIds)
        walkLabels.add(
            spec.getJSONObject("labels").let { labels ->
                val sb = StringBuilder()
                for (k in labels.keys()) sb.append(labels.getString(k)).append("、")
                sb.toString().trimEnd('、')
            }
        )
    }

    fun setMode(m: Int) {
        if (m in 0..1) mode = m
    }

    fun warmup() {
        run(IntArray(inputSize * inputSize) { -0x1000000 })
    }

    fun run(rgb: IntArray): SegResult {
        val t0 = System.nanoTime()
        val m = mode
        val session = sessions[m]
        fillTensor(rgb)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val walkable = ByteArray(maskSize * maskSize)
        var walkCount = 0
        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatBuf), shape).use { tensor ->
            session.run(mapOf(inputNames[m] to tensor)).use { out ->
                val logits = out[0].value
                val rows = toRows(logits, m)
                val n = maskSize * maskSize
                for (p in 0 until n) {
                    var bestVal = -1e9f
                    var bestId = -1
                    for (c in 0 until numClasses[m]) {
                        val v = rows[c][p]
                        if (v > bestVal) {
                            bestVal = v
                            bestId = c
                        }
                    }
                    if (walkSets[m].contains(bestId)) {
                        walkable[p] = 1
                        walkCount++
                    }
                }
            }
        }
        val t1 = System.nanoTime()
        return SegResult(walkable, maskSize, 100f * walkCount / walkable.size, (t1 - t0) / 1_000_000)
    }

    private fun fillTensor(rgb: IntArray) {
        val n = inputSize * inputSize
        for (i in 0 until n) {
            val p = rgb[i]
            floatBuf[i] = (((p shr 16) and 0xFF) / 255f - means[0]) / stds[0]
            floatBuf[n + i] = (((p shr 8) and 0xFF) / 255f - means[1]) / stds[1]
            floatBuf[2 * n + i] = ((p and 0xFF) / 255f - means[2]) / stds[2]
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun toRows(value: Any, m: Int): Array<FloatArray> {
        var node = value as Array<*>
        while (node.size == 1 && node[0] is Array<*> && (node[0] as Array<*>).size == numClasses[m]) {
            node = node[0] as Array<*>
        }
        val rows = Array(numClasses[m]) { FloatArray(maskSize * maskSize) }
        for (c in 0 until numClasses[m]) {
            val planes = node[c] as Array<*>
            var idx = 0
            for (y in 0 until maskSize) {
                val row = planes[y] as FloatArray
                System.arraycopy(row, 0, rows[c], idx, maskSize)
                idx += maskSize
            }
        }
        return rows
    }

    fun labelsFor(m: Int): String = walkLabels.getOrElse(m) { "" }

    fun close() {
        sessions.forEach { it.close() }
    }

    companion object {
        private const val TAG = "SegEngine"
    }
}
