package com.walkcam.app

import android.content.Context
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SegEngine(context: Context) {

    data class SegResult(
        val walkable: ByteArray,
        val maskSize: Int,
        val walkPct: Float,
        val ms: Long,
        val centerClass: String,
        val centerTop3: String
    )

    private var interpreter: Interpreter
    private val walkSet: Set<Int>
    private val allLabels: Map<Int, String>
    private val offset: Int
    private val inputPrep: String

    private val maskSize = 257
    private val numClasses = 151

    private val inBuf: ByteBuffer
    private val outBuf: ByteBuffer
    private val outIsLabel: Boolean
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
        offset = root.optInt("offset", 1)
        inputPrep = root.optString("input_prep", "div255")

        val model = loadModel(context)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(6) })
        val outShape = interpreter.getOutputTensor(0).shape()
        outIsLabel = outShape.size == 4 && outShape[3] == 1L
        inBuf = ByteBuffer.allocateDirect(maskSize * maskSize * 3 * 4).order(ByteOrder.nativeOrder())
        outBuf = ByteBuffer.allocateDirect(maskSize * maskSize * numClasses * 4).order(ByteOrder.nativeOrder())
    }

    private fun loadModel(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd("deeplab_ade.tflite")
        java.io.FileInputStream(fd.fileDescriptor).use { fis ->
            return fis.channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
    }

    fun warmup() {
        run(IntArray(maskSize * maskSize) { -0x1000000 })
    }

    fun run(rgb: IntArray): SegResult {
        val t0 = System.nanoTime()
        fillInput(rgb)
        inBuf.rewind()
        outBuf.rewind()
        interpreter.run(inBuf, outBuf)
        outBuf.rewind()

        val n = maskSize * maskSize
        val walk = ByteArray(n)
        for (p in 0 until n) {
            val id: Int = if (outIsLabel) {
                outBuf.getInt(p * 4).toInt()
            } else {
                var best = -1e9f
                var bestId = 0
                val base = p * numClasses
                for (c in 0 until numClasses) {
                    val v = outBuf.getFloat((base + c) * 4)
                    if (v > best) {
                        best = v
                        bestId = c
                    }
                }
                bestId
            }
            val segId = id - offset
            if (segId >= 0 && walkSet.contains(segId)) walk[p] = 1
        }
        val mask = majorityFilter(walk)
        var walkCount = 0
        for (b in mask) if (b.toInt() == 1) walkCount++
        val t1 = System.nanoTime()

        val centerId: Int = if (outIsLabel) {
            0
        } else {
            var best = -1e9f
            var bestId = 0
            val cp = (maskSize / 2 * maskSize + maskSize / 2) * numClasses
            for (c in 0 until numClasses) {
                val v = outBuf.getFloat((cp + c) * 4)
                if (v > best) {
                    best = v
                    bestId = c
                }
            }
            bestId - offset
        }
        val centerName = allLabels[centerId] ?: "?$centerId"
        val top3 = centerName
        return SegResult(mask, maskSize, 100f * walkCount / mask.size, (t1 - t0) / 1_000_000, centerName, top3)
    }

    private fun fillInput(rgb: IntArray) {
        val n = maskSize * maskSize
        val bytes = maskSize * maskSize * 3 * 4
        inBuf.clear()
        inBuf.limit(bytes)
        for (i in 0 until n) {
            val p = rgb[i]
            val r = ((p shr 16) and 0xFF).toFloat()
            val g = ((p shr 8) and 0xFF).toFloat()
            val b = (p and 0xFF).toFloat()
            when (inputPrep) {
                "mobilenet" -> {
                    inBuf.putFloat((r / 127.5f - 1f))
                    inBuf.putFloat((g / 127.5f - 1f))
                    inBuf.putFloat((b / 127.5f - 1f))
                }
                "raw" -> {
                    inBuf.putFloat(r)
                    inBuf.putFloat(g)
                    inBuf.putFloat(b)
                }
                else -> {
                    inBuf.putFloat(r / 255f)
                    inBuf.putFloat(g / 255f)
                    inBuf.putFloat(b / 255f)
                }
            }
        }
        inBuf.flip()
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
        interpreter.close()
    }
}
