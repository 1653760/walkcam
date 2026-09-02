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
    private val emaFloat = FloatArray(maskSize * maskSize)
    private var emaReady = false

    // EMA temporal smoothing
    private val emaAlpha = 0.7f
    private val emaThresh = 0.5f

    // Minimum connected-component size to keep (removes single-pixel noise only)
    private val minCompPixels = 10

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
        Log.i(TAG, "seg engine ready input=$inputName emaAlpha=$emaAlpha minComp=$minCompPixels")
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
        // Step 1: 3x3 majority filter (light spatial denoise)
        val maj = majorityFilter(raw)
        // Step 2: drop tiny isolated specks (< minCompPixels)
        val denoised = dropSmallComponents(maj)
        // Step 3: temporal EMA smoothing
        val ema = emaBlend(denoised)
        var walkCount = 0
        for (b in ema) if (b.toInt() == 1) walkCount++
        val t1 = System.nanoTime()
        val centerName = allLabels[centerId] ?: "?$centerId"
        return SegResult(ema, maskSize, 100f * walkCount / ema.size, (t1 - t0) / 1_000_000, centerName, centerName)
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
                    filtered[y * n + x] = raw[y * n + x]; continue
                }
                var count = 0
                for (dy in -1..1) for (dx in -1..1) {
                    if (raw[(y + dy) * n + (x + dx)].toInt() == 1) count++
                }
                filtered[y * n + x] = if (count >= 5) 1 else 0
            }
        }
        return filtered
    }

    /** Remove connected components smaller than minCompPixels (4-connectivity). */
    private fun dropSmallComponents(src: ByteArray): ByteArray {
        val n = maskSize
        val total = n * n
        val label = IntArray(total)
        val stack = IntArray(total)
        val out = ByteArray(total)
        var nextLabel = 0
        // store sizes indexed by label-1
        val sizes = ArrayList<Int>(64)
        for (start in 0 until total) {
            if (src[start].toInt() != 1 || label[start] != 0) continue
            nextLabel++
            var sp = 0; var size = 0
            stack[sp++] = start; label[start] = nextLabel
            while (sp > 0) {
                val cur = stack[--sp]; size++
                val x = cur % n; val y = cur / n
                if (x > 0     && src[cur-1].toInt()==1 && label[cur-1]==0) { label[cur-1]=nextLabel; stack[sp++]=cur-1 }
                if (x < n-1   && src[cur+1].toInt()==1 && label[cur+1]==0) { label[cur+1]=nextLabel; stack[sp++]=cur+1 }
                if (y > 0     && src[cur-n].toInt()==1 && label[cur-n]==0) { label[cur-n]=nextLabel; stack[sp++]=cur-n }
                if (y < n-1   && src[cur+n].toInt()==1 && label[cur+n]==0) { label[cur+n]=nextLabel; stack[sp++]=cur+n }
            }
            sizes.add(size)
        }
        for (i in 0 until total) {
            val lb = label[i]
            out[i] = if (lb > 0 && sizes[lb - 1] >= minCompPixels) 1 else 0
        }
        return out
    }

    /** EMA over binary mask stored as float in [0,1], re-binarized at emaThresh. */
    private fun emaBlend(src: ByteArray): ByteArray {
        val n = maskSize * maskSize
        if (!emaReady) {
            for (i in 0 until n) emaFloat[i] = src[i].toFloat()
            emaReady = true
        } else {
            val a = emaAlpha; val b = 1f - a
            for (i in 0 until n) emaFloat[i] = a * src[i].toFloat() + b * emaFloat[i]
        }
        val out = ByteArray(n)
        for (i in 0 until n) out[i] = if (emaFloat[i] >= emaThresh) 1 else 0
        return out
    }

    fun close() { session.close() }

    companion object { private const val TAG = "SegEngine" }
}
