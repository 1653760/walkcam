package com.walkcam.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.FloatBuffer

class SegEngine(context: Context) {

    data class SegResult(
        val walkable: ByteArray,
        val maskSize: Int,
        val walkPct: Float,
        val ms: Long,
        val centerClass: String
    )

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession
    private var inputName: String
    private val walkSet: Set<Int>
    val walkLabels: String
    private val allLabels: Map<Int, String>

    private val inputSize = 512
    private val maskSize = 128
    private val numClasses = 150
    private val floatBuf = FloatArray(3 * inputSize * inputSize)
    private val means = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val stds = floatArrayOf(0.229f, 0.224f, 0.225f)
    private val filtered = ByteArray(maskSize * maskSize)

    init {
        val spec = context.assets.open("walkable.json").bufferedReader().use { it.readText() }
        val root = JSONObject(spec)
        val bytes = context.assets.open("seg.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputInfo.keys.first()
        val arr = root.getJSONArray("walkable")
        val s = HashSet<Int>(arr.length())
        for (i in 0 until arr.length()) s.add(arr.getInt(i))
        walkSet = s
        walkLabels = root.getJSONObject("labels").let { labels ->
            val sb = StringBuilder()
            for (k in labels.keys()) sb.append(labels.getString(k)).append("、")
            sb.toString().trimEnd('、')
        }
        val allObj = root.getJSONObject("all")
        val al = HashMap<Int, String>(allObj.length())
        for (k in allObj.keys()) al[k.toInt()] = allObj.getString(k)
        allLabels = al
        Log.i(TAG, "seg engine ready, classes=$numClasses, walkable=${s.size}")
    }

    fun warmup() {
        run(IntArray(inputSize * inputSize) { -0x1000000 })
    }

    fun run(rgb: IntArray): SegResult {
        val t0 = System.nanoTime()
        fillTensor(rgb)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val raw = ByteArray(maskSize * maskSize)
        val classMask = ByteArray(maskSize * maskSize)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatBuf), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { out ->
                val rows = toRows(out[0].value)
                val n = maskSize * maskSize
                for (p in 0 until n) {
                    var bestVal = -1e9f
                    var bestId = -1
                    for (c in 0 until numClasses) {
                        val v = rows[c][p]
                        if (v > bestVal) {
                            bestVal = v
                            bestId = c
                        }
                    }
                    classMask[p] = bestId.toByte()
                    raw[p] = if (walkSet.contains(bestId)) 1 else 0
                }
            }
        }
        val centerId = classMask[(maskSize / 2) * maskSize + maskSize / 2].toInt() and 0xFF
        val centerClass = allLabels[centerId] ?: "?$centerId"
        val mask = majorityFilter(raw)
        var walkCount = 0
        for (b in mask) if (b.toInt() == 1) walkCount++
        val t1 = System.nanoTime()
        return SegResult(mask, maskSize, 100f * walkCount / mask.size, (t1 - t0) / 1_000_000, centerClass)
    }

    private fun majorityFilter(raw: ByteArray): ByteArray {
        val n = maskSize
        for (y in 0 until n) {
            for (x in 0 until n) {
                if (x == 0 || y == 0 || x == n - 1 || y == n - 1) {
                    filtered[y * n + x] = raw[y * n + x]
                    continue
                }
                var count = 0
                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (raw[(y + dy) * n + (x + dx)].toInt() == 1) count++
                    }
                }
                filtered[y * n + x] = if (count >= 5) 1 else 0
            }
        }
        return filtered.copyOf()
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
    private fun toRows(value: Any): Array<FloatArray> {
        var node = value as Array<*>
        while (node.size == 1 && node[0] is Array<*> && (node[0] as Array<*>).size == numClasses) {
            node = node[0] as Array<*>
        }
        val rows = Array(numClasses) { FloatArray(maskSize * maskSize) }
        for (c in 0 until numClasses) {
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

    fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "SegEngine"
    }
}
