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
        val centerClass: String,
        val centerTop3: String
    )

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession
    private var inputName: String
    private val walkSet: Set<Int>
    private val allLabels: Map<Int, String>

    private val inputSize = 512
    private val maskSize = 128
    private val floatBuf = FloatArray(3 * inputSize * inputSize)
    private val filtered = ByteArray(maskSize * maskSize)

    init {
        val spec = context.assets.open("walkable.json").bufferedReader().use { it.readText() }
        val root = JSONObject(spec)
        val arr = root.getJSONArray("walkable")
        val s = HashSet<Int>(arr.length())
        for (i in 0 until arr.length()) s.add(arr.getInt(i))
        walkSet = s
        val allObj = root.getJSONObject("all")
        val al = HashMap<Int, String>(allObj.length())
        for (k in allObj.keys()) al[k.toInt()] = allObj.getString(k)
        allLabels = al

        val bytes = context.assets.open("seg.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(6)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputInfo.keys.first()
        Log.i(TAG, "seg engine ready, input=$inputName")
    }

    fun warmup() {
        run(IntArray(inputSize * inputSize) { -0x1000000 })
    }

    fun run(rgb: IntArray): SegResult {
        val t0 = System.nanoTime()
        fillTensor(rgb)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val raw = ByteArray(maskSize * maskSize)
        var centerId = -1
        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatBuf), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { out ->
                val value = out[0].value
                val rows = toRows(value)
                for (y in 0 until maskSize) {
                    val row = rows[y]
                    for (x in 0 until maskSize) {
                        val id = row[x].toInt()
                        if (walkSet.contains(id)) raw[y * maskSize + x] = 1
                    }
                }
                centerId = rows[maskSize / 2][maskSize / 2].toInt()
            }
        }
        val mask = majorityFilter(raw)
        var walkCount = 0
        for (b in mask) if (b.toInt() == 1) walkCount++
        val t1 = System.nanoTime()
        val centerName = allLabels[centerId] ?: "?$centerId"
        return SegResult(mask, maskSize, 100f * walkCount / mask.size, (t1 - t0) / 1_000_000, centerName, centerName)
    }

    private fun fillTensor(rgb: IntArray) {
        val n = inputSize * inputSize
        for (i in 0 until n) {
            val p = rgb[i]
            floatBuf[i] = ((p shr 16) and 0xFF).toFloat()
            floatBuf[n + i] = ((p shr 8) and 0xFF).toFloat()
            floatBuf[2 * n + i] = (p and 0xFF).toFloat()
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun toRows(value: Any): Array<LongArray> {
        var node = value as Array<*>
        while (node.size == 1 && node[0] is Array<*> && (node[0] as Array<*>).size == maskSize) {
            node = node[0] as Array<*>
        }
        return node as Array<LongArray>
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

    fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "SegEngine"
    }
}
