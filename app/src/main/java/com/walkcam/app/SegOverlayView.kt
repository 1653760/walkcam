package com.walkcam.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class SegOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private var walkable: ByteArray? = null
    private var maskSize = 128
    private var fullFrame = true   // true = full-frame stretch mode (no crop rect)
    private var maskBitmap: Bitmap? = null

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x5900E676.toInt()
    }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = 0xFF00E676.toInt()
    }
    private val nonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFFFF5252.toInt()
        textSize = 46f
        isFakeBoldText = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    private var debugRgb: IntArray? = null
    private var debugCenterClass = ""
    private var debugBitmap: Bitmap? = null
    private val debugPaint = Paint().apply { isFilterBitmap = true }
    private val debugBorderPaint = Paint().apply {
        style = Paint.Style.STROKE; strokeWidth = 4f; color = Color.YELLOW
    }
    private val debugTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW; textSize = 34f; isFakeBoldText = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    fun setDebug(rgb: IntArray?, centerClass: String) {
        debugRgb = rgb
        debugCenterClass = centerClass
    }

    fun update(walkable: ByteArray, maskSize: Int, info: YuvToRgb.FrameInfo) {
        this.walkable = walkable
        this.maskSize = maskSize
        // cropSize == -1 means full-frame stretch mode
        this.fullFrame = info.cropSize < 0
        invalidate()
    }

    fun clear() { walkable = null; invalidate() }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val mask = walkable ?: return
        val vw = width.toFloat()
        val vh = height.toFloat()

        // In full-frame mode the mask covers the entire view.
        val drawRect = RectF(0f, 0f, vw, vh)

        // Upsample mask 2× (nearest-neighbour) -> upSize×upSize for finer contours
        val upSize = maskSize * 2
        val upMask = upsample2x(mask, maskSize)

        val polys = extractPolygons(upMask, upSize)
        if (polys.isEmpty()) {
            canvas.drawText("前方无可通行区域", drawRect.left + 20f, drawRect.top + drawRect.height() / 2f, nonePaint)
        }
        for (poly in polys) {
            if (poly.size < 3) continue
            val path = Path()
            for (i in poly.indices) {
                // poly coords are in upSize space; map to view
                val sx = drawRect.left + poly[i][0] / upSize * drawRect.width()
                val sy = drawRect.top  + poly[i][1] / upSize * drawRect.height()
                if (i == 0) path.moveTo(sx, sy) else path.lineTo(sx, sy)
            }
            path.close()
            canvas.drawPath(path, fillPaint)
            canvas.drawPath(path, strokePaint)
        }

        // Debug thumbnail (top-right)
        val dbg = debugRgb
        if (dbg != null && dbg.isNotEmpty()) {
            val sideLen = Math.sqrt(dbg.size.toDouble()).toInt()
            if (sideLen * sideLen == dbg.size) {
                var db = debugBitmap
                if (db == null || db.width != sideLen) {
                    db = Bitmap.createBitmap(sideLen, sideLen, Bitmap.Config.ARGB_8888)
                    debugBitmap = db
                }
                db.setPixels(dbg, 0, sideLen, 0, 0, sideLen, sideLen)
                val side = 200f
                val dx = width - side - 14f; val dy = 14f
                val dst = RectF(dx, dy, dx + side, dy + side)
                canvas.drawBitmap(db, null, dst, debugPaint)
                canvas.drawRect(dst, debugBorderPaint)
                canvas.drawText("中心:$debugCenterClass", dx, dy + side + 40f, debugTextPaint)
            }
        }
    }

    /** 2× nearest-neighbour upsample: n×n -> (2n)×(2n) */
    private fun upsample2x(src: ByteArray, n: Int): ByteArray {
        val m = n * 2
        val dst = ByteArray(m * m)
        for (y in 0 until n) {
            for (x in 0 until n) {
                val v = src[y * n + x]
                val dy = y * 2; val dx = x * 2
                dst[dy * m + dx]         = v
                dst[dy * m + dx + 1]     = v
                dst[(dy+1) * m + dx]     = v
                dst[(dy+1) * m + dx + 1] = v
            }
        }
        return dst
    }

    private fun extractPolygons(mask: ByteArray, n: Int): List<List<FloatArray>> {
        val comps = connectedComponents(mask, n)
        if (comps.isEmpty()) return emptyList()
        val biggest = comps[0].size
        // Keep components >= 1% of biggest OR >= 30px; show up to 6 regions
        val minSize = maxOf(30, biggest / 100)
        val out = ArrayList<List<FloatArray>>(6)
        for (comp in comps) {
            if (comp.size < minSize || out.size >= 6) break
            val boundary = traceBoundary(comp, n) ?: continue
            val smoothed = smoothClosed(boundary, 5, 2)
            val poly = simplify(smoothed, n * 0.02f)   // tighter epsilon -> more detail
            if (poly.size >= 3) out.add(chaikin(poly, 2))
        }
        return out
    }

    private fun smoothClosed(pts: ArrayList<FloatArray>, window: Int, passes: Int): ArrayList<FloatArray> {
        var cur = pts
        repeat(passes) {
            val m = cur.size
            if (m < window + 2) return cur
            val half = window / 2
            val out = ArrayList<FloatArray>(m)
            for (i in 0 until m) {
                var sx = 0f; var sy = 0f; var c = 0
                for (k in -half..half) {
                    val j = ((i + k) % m + m) % m
                    sx += cur[j][0]; sy += cur[j][1]; c++
                }
                out.add(floatArrayOf(sx / c, sy / c))
            }
            cur = out
        }
        return cur
    }

    private fun chaikin(pts: List<FloatArray>, iterations: Int): ArrayList<FloatArray> {
        var cur: List<FloatArray> = pts
        for (iter in 0 until iterations) {
            val m = cur.size; if (m < 3) break
            val out = ArrayList<FloatArray>(m * 2)
            for (i in 0 until m) {
                val a = cur[i]; val b = cur[(i + 1) % m]
                out.add(floatArrayOf(a[0]*0.75f + b[0]*0.25f, a[1]*0.75f + b[1]*0.25f))
                out.add(floatArrayOf(a[0]*0.25f + b[0]*0.75f, a[1]*0.25f + b[1]*0.75f))
            }
            cur = out
        }
        return ArrayList(cur)
    }

    private fun traceBoundary(comp: IntArray, n: Int): ArrayList<FloatArray>? {
        val inComp = BooleanArray(n * n)
        var minIdx = Int.MAX_VALUE
        for (idx in comp) { inComp[idx] = true; if (idx < minIdx) minIdx = idx }
        val dxs = intArrayOf(1,1,0,-1,-1,-1,0,1)
        val dys = intArrayOf(0,1,1,1,0,-1,-1,-1)
        var px = minIdx % n; var py = minIdx / n
        var back = 4
        val pts = ArrayList<FloatArray>(512)
        pts.add(floatArrayOf(px.toFloat(), py.toFloat()))
        val startX = px; val startY = py
        val maxSteps = comp.size * 8 + 64
        var steps = 0; var firstMove = true; var firstDir = -1
        while (steps++ < maxSteps) {
            var found = false
            for (k in 1..8) {
                val d = (back + k) % 8
                val nx = px + dxs[d]; val ny = py + dys[d]
                if (nx in 0 until n && ny in 0 until n && inComp[ny * n + nx]) {
                    if (firstMove) { firstDir = d; firstMove = false }
                    else if (nx == startX && ny == startY && d == firstDir) return pts
                    pts.add(floatArrayOf(nx.toFloat(), ny.toFloat()))
                    back = (d + 4) % 8; px = nx; py = ny; found = true; break
                }
            }
            if (!found) return if (pts.size >= 3) pts else null
        }
        return if (pts.size >= 3) pts else null
    }

    private fun connectedComponents(mask: ByteArray, n: Int): List<IntArray> {
        val total = n * n
        val seen = BooleanArray(total)
        val comps = ArrayList<IntArray>()
        val stack = IntArray(total)
        for (start in 0 until total) {
            if (seen[start] || mask[start].toInt() != 1) continue
            var sp = 0
            val comp = ArrayList<Int>(256)
            stack[sp++] = start; seen[start] = true
            while (sp > 0) {
                val cur = stack[--sp]; comp.add(cur)
                val x = cur % n; val y = cur / n
                if (x > 0   && !seen[cur-1] && mask[cur-1].toInt()==1) { seen[cur-1]=true; stack[sp++]=cur-1 }
                if (x < n-1 && !seen[cur+1] && mask[cur+1].toInt()==1) { seen[cur+1]=true; stack[sp++]=cur+1 }
                if (y > 0   && !seen[cur-n] && mask[cur-n].toInt()==1) { seen[cur-n]=true; stack[sp++]=cur-n }
                if (y < n-1 && !seen[cur+n] && mask[cur+n].toInt()==1) { seen[cur+n]=true; stack[sp++]=cur+n }
            }
            comps.add(comp.toIntArray())
        }
        comps.sortByDescending { it.size }
        return comps
    }

    private fun simplify(pts: List<FloatArray>, eps: Float): ArrayList<FloatArray> {
        if (pts.size <= 4) return ArrayList(pts)
        val keep = BooleanArray(pts.size) { true }
        simplifyRec(pts, 0, pts.size - 1, eps, keep)
        val out = ArrayList<FloatArray>()
        for (i in pts.indices) if (keep[i]) out.add(pts[i])
        return out
    }

    private fun simplifyRec(pts: List<FloatArray>, first: Int, last: Int, eps: Float, keep: BooleanArray) {
        if (last <= first + 1) return
        var maxDist = 0f; var idx = -1
        val a = pts[first]; val b = pts[last]
        val dx = b[0]-a[0]; val dy = b[1]-a[1]
        val len = Math.sqrt((dx*dx+dy*dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
        for (i in first+1 until last) {
            val p = pts[i]
            val d = Math.abs((p[0]-a[0])*dy - (p[1]-a[1])*dx) / len
            if (d > maxDist) { maxDist = d; idx = i }
        }
        if (maxDist > eps && idx > 0) {
            keep[idx] = true
            simplifyRec(pts, first, idx, eps, keep)
            simplifyRec(pts, idx, last, eps, keep)
        }
    }
}
