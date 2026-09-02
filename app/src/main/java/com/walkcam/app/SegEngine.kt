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
    private val opened = ByteArray(maskSize * maskSize)
    private val tmpMask = ByteArray(maskSize * maskSize)
    private val ccLabel = IntArray(maskSize * maskSize)
    private val ccStack = IntArray(maskSize * maskSize)
    private val emaFloat = FloatArray(maskSize * maskSize)
    private var emaReady = false

    // v0.14 post-processing knobs
    // Hard vertical prior: walking-path pixels almost never appear in the upper third of the frame.
    // Everything with y < skyRowCutoff is forced to non-walkable BEFORE connectivity analysis.
    private val skyRowCutoff = maskSize / 3   // 42 for size 128 -> top ~1/3 zeroed
    // Bottom anchor: any kept component must touch y >= anchorY. Relaxed from 96 to 85 (~lower 1/3).
    private val anchorY = (maskSize * 2) / 3   // 85 for size 128
    // EMA blending: newMask * alpha + prev * (1-alpha). Raised for faster response in clutter.
    private val emaAlpha = 0.75f
    // Binarize threshold on EMA float mask
    private val emaThresh = 0.5f
    // Min component size (pixels) to keep. Components smaller than this are dropped even if bottom-anchored.
    private val minCompPixels = 25

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
        Log.i(TAG, "seg engine ready, input=$inputName anchorY=$anchorY emaAlpha=$emaAlpha")
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
        // Step 1: hard vertical prior -- zero out the upper 1/3 (sky/wall/ceiling region)
        zeroTopRows(raw)
        // Step 2: 3x3 majority (spatial denoise on argmax)
        val maj = majorityFilter(raw)
        // Step 3: morphological opening (erode then dilate, 3x3) -- kills small wall bumps
        val op = openMask(maj)
        // Step 4: keep ALL connected components that touch the bottom anchor row and are big enough.
        //         This tolerates clutter that splits the floor into multiple regions.
        val cc = keepBottomAnchoredComponents(op)
        // Step 5: temporal EMA
        val ema = emaBlend(cc)
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
        return filtered
    }

    /** 3x3 morphological opening: erode then dilate. Kills small isolated blobs and thin bumps. */
    private fun openMask(src: ByteArray): ByteArray {
        val n = maskSize
        // erode -> tmpMask
        for (y in 0 until n) {
            for (x in 0 until n) {
                if (src[y * n + x].toInt() != 1) { tmpMask[y * n + x] = 0; continue }
                if (x == 0 || y == 0 || x == n - 1 || y == n - 1) { tmpMask[y * n + x] = 1; continue }
                var allOne = true
                loop@ for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (src[(y + dy) * n + (x + dx)].toInt() != 1) { allOne = false; break@loop }
                    }
                }
                tmpMask[y * n + x] = if (allOne) 1 else 0
            }
        }
        // dilate tmpMask -> opened
        for (y in 0 until n) {
            for (x in 0 until n) {
                var anyOne = false
                val y0 = if (y == 0) 0 else -1
                val y1 = if (y == n - 1) 0 else 1
                val x0 = if (x == 0) 0 else -1
                val x1 = if (x == n - 1) 0 else 1
                loop@ for (dy in y0..y1) {
                    for (dx in x0..x1) {
                        if (tmpMask[(y + dy) * n + (x + dx)].toInt() == 1) { anyOne = true; break@loop }
                    }
                }
                opened[y * n + x] = if (anyOne) 1 else 0
            }
        }
        return opened
    }

    /** Force upper 1/3 rows to non-walkable (sky/wall/ceiling hard prior). */
    private fun zeroTopRows(mask: ByteArray) {
        val cutoff = skyRowCutoff * maskSize
        for (i in 0 until cutoff) mask[i] = 0
    }

    /**
     * Iterative 4-connectivity flood fill. Keeps ALL components that (a) touch y >= anchorY
     * and (b) have size >= minCompPixels. This tolerates clutter that splits the floor into
     * several regions -- unlike a strict largest-only rule which would drop valid floor patches.
     * Components not anchored to the bottom (isolated wall/table false positives) are dropped.
     */
    private fun keepBottomAnchoredComponents(src: ByteArray): ByteArray {
        val n = maskSize
        java.util.Arrays.fill(ccLabel, 0)
        var nextLabel = 0
        // component metadata; nextLabel is 1-based, index by (label-1)
        val maxLabels = 4096
        val compTouches = BooleanArray(maxLabels)
        val compSize = IntArray(maxLabels)
        for (y in 0 until n) {
            for (x in 0 until n) {
                val idx0 = y * n + x
                if (src[idx0].toInt() != 1 || ccLabel[idx0] != 0) continue
                nextLabel++
                if (nextLabel >= maxLabels) break
                var top = 0
                ccStack[top++] = idx0
                ccLabel[idx0] = nextLabel
                var size = 0
                var touchesBottom = false
                while (top > 0) {
                    val idx = ccStack[--top]
                    size++
                    val cy = idx / n
                    val cx = idx - cy * n
                    if (cy >= anchorY) touchesBottom = true
                    if (cx > 0) {
                        val ni = idx - 1
                        if (src[ni].toInt() == 1 && ccLabel[ni] == 0) { ccLabel[ni] = nextLabel; ccStack[top++] = ni }
                    }
                    if (cx < n - 1) {
                        val ni = idx + 1
                        if (src[ni].toInt() == 1 && ccLabel[ni] == 0) { ccLabel[ni] = nextLabel; ccStack[top++] = ni }
                    }
                    if (cy > 0) {
                        val ni = idx - n
                        if (src[ni].toInt() == 1 && ccLabel[ni] == 0) { ccLabel[ni] = nextLabel; ccStack[top++] = ni }
                    }
                    if (cy < n - 1) {
                        val ni = idx + n
                        if (src[ni].toInt() == 1 && ccLabel[ni] == 0) { ccLabel[ni] = nextLabel; ccStack[top++] = ni }
                    }
                }
                compTouches[nextLabel - 1] = touchesBottom
                compSize[nextLabel - 1] = size
            }
        }
        for (i in 0 until n * n) {
            val lb = ccLabel[i]
            tmpMask[i] = if (lb > 0 && lb <= nextLabel && compTouches[lb - 1] && compSize[lb - 1] >= minCompPixels) 1 else 0
        }
        return tmpMask
    }

    /** EMA over binary mask stored as float in [0,1], re-binarized with emaThresh. */
    private fun emaBlend(src: ByteArray): ByteArray {
        val n = maskSize * maskSize
        if (!emaReady) {
            for (i in 0 until n) emaFloat[i] = src[i].toFloat()
            emaReady = true
        } else {
            val a = emaAlpha
            val b = 1f - a
            for (i in 0 until n) emaFloat[i] = a * src[i].toFloat() + b * emaFloat[i]
        }
        val out = ByteArray(n)
        for (i in 0 until n) out[i] = if (emaFloat[i] >= emaThresh) 1 else 0
        return out
    }

    fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "SegEngine"
    }
}
