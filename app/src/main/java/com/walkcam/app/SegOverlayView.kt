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
    private var srcW = 1
    private var srcH = 1
    private var cropSize = 1
    private var offX = 0
    private var offY = 0
    private var maskBitmap: Bitmap? = null
    private val maskColors = IntArray(128 * 128)

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = 0x5900E676.toInt()
    }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 7f
        color = 0xFF00E676.toInt()
    }
    private val borderPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 3f
        color = 0xAAFFFFFF.toInt()
    }
    private val legendPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = 0xFF00E676.toInt()
        textSize = 42f
        isFakeBoldText = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
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
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.YELLOW
    }
    private val debugTextPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.YELLOW
        textSize = 34f
        isFakeBoldText = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }

    fun setDebug(rgb: IntArray?, centerClass: String) {
        debugRgb = rgb
        debugCenterClass = centerClass
    }

    fun update(walkable: ByteArray, maskSize: Int, info: YuvToRgb.FrameInfo) {
        this.walkable = walkable
        this.maskSize = maskSize
        this.srcW = maxOf(1, info.w)
        this.srcH = maxOf(1, info.h)
        this.cropSize = maxOf(1, info.cropSize)
        this.offX = info.offX
        this.offY = info.offY
        invalidate()
    }

    fun clear() {
        walkable = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val mask = walkable ?: return

        val vw = width.toFloat()
        val vh = height.toFloat()
        val s = maxOf(vw / srcW, vh / srcH)
        val squareSide = cropSize * s
        val cxFrame = (offX + cropSize / 2f) * s + (vw - srcW * s) / 2f
        val cyFrame = (offY + cropSize / 2f) * s + (vh - srcH * s) / 2f
        val rect = RectF(
            cxFrame - squareSide / 2f,
            cyFrame - squareSide / 2f,
            cxFrame + squareSide / 2f,
            cyFrame + squareSide / 2f
        )
        canvas.drawRect(rect, borderPaint)

        val polys = extractPolygons(mask, maskSize)
        if (polys.isEmpty()) {
            canvas.drawText("前方无可通行区域", rect.left + 20f, rect.top + rect.height() / 2f, nonePaint)
        }
        for (poly in polys) {
            if (poly.size < 3) continue
            val path = Path()
            for (i in poly.indices) {
                val sx = rect.left + poly[i][0] / maskSize * rect.width()
                val sy = rect.top + poly[i][1] / maskSize * rect.height()
                if (i == 0) path.moveTo(sx, sy) else path.lineTo(sx, sy)
            }
            path.close()
            canvas.drawPath(path, fillPaint)
            canvas.drawPath(path, strokePaint)
        }
        legendPaint.color = 0xFF00E676.toInt()
        canvas.drawText("绿框内=可通行", rect.left + 10f, rect.top + 46f, legendPaint)

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
                val dx = width - side - 14f
                val dy = 14f
                val dst = RectF(dx, dy, dx + side, dy + side)
                canvas.drawBitmap(db, null, dst, debugPaint)
                canvas.drawRect(dst, debugBorderPaint)
                canvas.drawText("中心:$debugCenterClass", dx, dy + side + 40f, debugTextPaint)
            }
        }
    }

    private fun extractPolygons(mask: ByteArray, n: Int): List<List<FloatArray>> {
        val comps = connectedComponents(mask, n)
        if (comps.isEmpty()) return emptyList()
        val biggest = comps[0].size
        val minSize = maxOf(60, biggest / 5)
        val out = ArrayList<List<FloatArray>>()
        for (comp in comps) {
            if (comp.size < minSize || out.size >= 3) break
            val boundary = traceBoundary(comp, n) ?: continue
            val poly = simplify(boundary, n * 0.03f)
            if (poly.size >= 3) out.add(poly)
        }
        return out
    }

    private fun traceBoundary(comp: IntArray, n: Int): ArrayList<FloatArray>? {
        val inComp = BooleanArray(n * n)
        var minIdx = Int.MAX_VALUE
        for (idx in comp) {
            inComp[idx] = true
            if (idx < minIdx) minIdx = idx
        }
        val dxs = intArrayOf(1, 1, 0, -1, -1, -1, 0, 1)
        val dys = intArrayOf(0, 1, 1, 1, 0, -1, -1, -1)

        var px = minIdx % n
        var py = minIdx / n
        var back = 4
        val pts = ArrayList<FloatArray>(256)
        pts.add(floatArrayOf(px.toFloat(), py.toFloat()))
        val startX = px
        val startY = py
        val maxSteps = comp.size * 8 + 64
        var steps = 0
        var firstMove = true
        var firstDir = -1
        while (steps++ < maxSteps) {
            var found = false
            for (k in 1..8) {
                val d = (back + k) % 8
                val nx = px + dxs[d]
                val ny = py + dys[d]
                if (nx in 0 until n && ny in 0 until n && inComp[ny * n + nx]) {
                    if (firstMove) {
                        firstDir = d
                        firstMove = false
                    } else if (nx == startX && ny == startY && d == firstDir) {
                        return pts
                    }
                    pts.add(floatArrayOf(nx.toFloat(), ny.toFloat()))
                    back = (d + 4) % 8
                    px = nx
                    py = ny
                    found = true
                    break
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
            stack[sp++] = start
            seen[start] = true
            while (sp > 0) {
                val cur = stack[--sp]
                comp.add(cur)
                val x = cur % n
                val y = cur / n
                if (x > 0 && !seen[cur - 1] && mask[cur - 1].toInt() == 1) {
                    seen[cur - 1] = true; stack[sp++] = cur - 1
                }
                if (x < n - 1 && !seen[cur + 1] && mask[cur + 1].toInt() == 1) {
                    seen[cur + 1] = true; stack[sp++] = cur + 1
                }
                if (y > 0 && !seen[cur - n] && mask[cur - n].toInt() == 1) {
                    seen[cur - n] = true; stack[sp++] = cur - n
                }
                if (y < n - 1 && !seen[cur + n] && mask[cur + n].toInt() == 1) {
                    seen[cur + n] = true; stack[sp++] = cur + n
                }
            }
            comps.add(comp.toIntArray())
        }
        comps.sortByDescending { it.size }
        return comps
    }

    private fun cross(o: FloatArray, a: FloatArray, b: FloatArray): Float =
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    private fun convexHull(pts: ArrayList<FloatArray>): ArrayList<FloatArray> {
        pts.sortWith(compareBy({ it[0] }, { it[1] }))
        val hull = ArrayList<FloatArray>(pts.size / 2 + 4)
        for (p in pts) {
            while (hull.size >= 2 && cross(hull[hull.size - 2], hull[hull.size - 1], p) <= 0f) {
                hull.removeAt(hull.size - 1)
            }
            hull.add(p)
        }
        val lowerSize = hull.size + 1
        for (i in pts.size - 2 downTo 0) {
            val p = pts[i]
            while (hull.size >= lowerSize && cross(hull[hull.size - 2], hull[hull.size - 1], p) <= 0f) {
                hull.removeAt(hull.size - 1)
            }
            hull.add(p)
        }
        if (hull.size > 1) hull.removeAt(hull.size - 1)
        return hull
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
        var maxDist = 0f
        var idx = -1
        val a = pts[first]
        val b = pts[last]
        val dx = b[0] - a[0]
        val dy = b[1] - a[1]
        val len = Math.sqrt((dx * dx + dy * dy).toDouble()).toFloat().coerceAtLeast(1e-6f)
        for (i in first + 1 until last) {
            val p = pts[i]
            val d = Math.abs((p[0] - a[0]) * dy - (p[1] - a[1]) * dx) / len
            if (d > maxDist) {
                maxDist = d
                idx = i
            }
        }
        if (maxDist > eps && idx > 0) {
            keep[idx] = true
            simplifyRec(pts, first, idx, eps, keep)
            simplifyRec(pts, idx, last, eps, keep)
        }
    }
}
