package com.walkcam.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

class SegEngine(context: Context) {

    data class SegResult(
        val walkable: ByteArray,
        val maskSize: Int,
        val walkPct: Float,
        val ms: Long,
        val centerClass: String,
        val centerTop3: String
    )

    private data class Det(val ci: Int, val score: Float, val cx: Float, val cy: Float, val w: Float, val h: Float, val anchor: Int)

    private val env = OrtEnvironment.getEnvironment()
    private var session: OrtSession
    private var inputName: String
    private val labels: List<String>

    private val inputSize = 640
    private val numAnchors = 8400
    private val numClasses = 12
    private val numCoeffs = 32
    private val protoSize = 160
    private val confThreshold = 0.35f
    private val iouThreshold = 0.45f

    private val floatBuf = FloatArray(3 * inputSize * inputSize)
    private val coeffBuf = FloatArray(numCoeffs)
    private val filtered = ByteArray(protoSize * protoSize)

    init {
        val spec = context.assets.open("walkable.json").bufferedReader().use { it.readText() }
        val root = JSONObject(spec)
        val labelsObj = root.getJSONObject("all")
        val l = ArrayList<String>(numClasses)
        for (i in 0 until numClasses) l.add(labelsObj.optString(str(i), "?$i"))
        labels = l

        val bytes = context.assets.open("seg.onnx").readBytes()
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(6)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(bytes, opts)
        inputName = session.inputInfo.keys.first()
        Log.i(TAG, "seg engine ready, input=$inputName")
    }

    private fun str(i: Int) = i.toString()

    fun warmup() {
        run(IntArray(inputSize * inputSize) { -0x1000000 })
    }

    fun run(rgb: IntArray): SegResult {
        val t0 = System.nanoTime()
        fillTensor(rgb)
        val shape = longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        val walk = ByteArray(protoSize * protoSize)
        OnnxTensor.createTensor(env, FloatBuffer.wrap(floatBuf), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { out ->
                val rows0 = strip(out[0].value, 4 + numClasses + numCoeffs)
                val protos = flatten2d(out[1].value, numCoeffs)

                var cands = ArrayList<Det>()
                for (a in 0 until numAnchors) {
                    var best = 0f
                    var ci = -1
                    for (k in 0 until numClasses) {
                        val s = rows0[4 + k][a]
                        if (s > best) {
                            best = s
                            ci = k
                        }
                    }
                    if (best >= confThreshold) {
                        cands.add(Det(ci, best, rows0[0][a], rows0[1][a], rows0[2][a], rows0[3][a], a))
                    }
                }
                cands = nms(cands) as ArrayList<Det>

                for (d in cands.take(25)) {
                    for (k in 0 until numCoeffs) {
                        coeffBuf[k] = rows0[4 + numClasses + k][d.anchor]
                    }
                    val scale = protoSize.toFloat() / inputSize
                    val bx1 = max(0, ((d.cx - d.w / 2f) * scale).toInt())
                    val by1 = max(0, ((d.cy - d.h / 2f) * scale).toInt())
                    val bx2 = min(protoSize - 1, ((d.cx + d.w / 2f) * scale).toInt())
                    val by2 = min(protoSize - 1, ((d.cy + d.h / 2f) * scale).toInt())
                    for (y in by1..by2) {
                        val rowBase = y * protoSize
                        for (x in bx1..bx2) {
                            if (walk[rowBase + x].toInt() == 1) continue
                            var m = 0f
                            for (k in 0 until numCoeffs) {
                                m += coeffBuf[k] * protos[k][rowBase + x]
                            }
                            val sig = 1f / (1f + exp(-m))
                            if (sig > 0.55f) walk[rowBase + x] = 1
                        }
                    }
                }
            }
        }
        val mask = majorityFilter(walk)
        var walkCount = 0
        for (b in mask) if (b.toInt() == 1) walkCount++
        val t1 = System.nanoTime()

        val top3 = lastTop3
        val top3Str = if (top3.isEmpty()) "无检出" else top3.joinToString(" / ") { "${labels[it.ci]} ${"%.2f".format(it.score)}" }
        return SegResult(
            mask, protoSize, 100f * walkCount / mask.size, (t1 - t0) / 1_000_000,
            top3.firstOrNull()?.let { labels[it.ci] } ?: "无检出", top3Str
        )
    }

    private var lastTop3: List<Det> = emptyList()

    private fun nms(cands: List<Det>): List<Det> {
        val byClass = cands.groupBy { it.ci }
        val keptAll = ArrayList<Det>()
        for ((_, dets) in byClass) {
            val sorted = dets.sortedByDescending { it.score }
            val keep = BooleanArray(sorted.size) { true }
            for (i in sorted.indices) {
                if (!keep[i]) continue
                for (j in i + 1 until sorted.size) {
                    if (!keep[j]) continue
                    if (iouBox(sorted[i], sorted[j]) > iouThreshold) keep[j] = false
                }
            }
            for (i in sorted.indices) if (keep[i]) keptAll.add(sorted[i])
        }
        lastTop3 = keptAll.sortedByDescending { it.score }.take(3)
        return keptAll
    }

    private fun iouBox(a: Det, b: Det): Float {
        val ax1 = a.cx - a.w / 2f
        val ay1 = a.cy - a.h / 2f
        val ax2 = a.cx + a.w / 2f
        val ay2 = a.cy + a.h / 2f
        val bx1 = b.cx - b.w / 2f
        val by1 = b.cy - b.h / 2f
        val bx2 = b.cx + b.w / 2f
        val by2 = b.cy + b.h / 2f
        val ix = max(0f, min(ax2, bx2) - max(ax1, bx1))
        val iy = max(0f, min(ay2, by2) - max(ay1, by1))
        val inter = ix * iy
        val u = a.w * a.h + b.w * b.h - inter
        return if (u > 0) inter / u else 0f
    }

    private fun majorityFilter(raw: ByteArray): ByteArray {
        val n = protoSize
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
            floatBuf[i] = ((p shr 16) and 0xFF) / 255f
            floatBuf[n + i] = ((p shr 8) and 0xFF) / 255f
            floatBuf[2 * n + i] = (p and 0xFF) / 255f
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun strip(value: Any, expectRows: Int): Array<FloatArray> {
        var node = value as Array<*>
        while (node.size == 1 && node[0] is Array<*> && (node[0] as Array<*>).size == expectRows) {
            node = node[0] as Array<*>
        }
        return node as Array<FloatArray>
    }

    @Suppress("UNCHECKED_CAST")
    private fun flatten2d(value: Any, channels: Int): Array<FloatArray> {
        var node = value as Array<*>
        while (node.size == 1 && node[0] is Array<*> && (node[0] as Array<*>).size == channels) {
            node = node[0] as Array<*>
        }
        val out = Array(channels) { FloatArray(protoSize * protoSize) }
        for (c in 0 until channels) {
            val planes = node[c] as Array<*>
            var idx = 0
            for (y in 0 until protoSize) {
                val row = planes[y] as FloatArray
                System.arraycopy(row, 0, out[c], idx, protoSize)
                idx += protoSize
            }
        }
        return out
    }

    fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "SegEngine"
    }
}
