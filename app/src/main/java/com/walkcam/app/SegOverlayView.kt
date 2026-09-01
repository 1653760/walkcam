package com.walkcam.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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

    private val edgePaint = Paint().apply { color = Color.WHITE }
    private val legendPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 40f
        isFakeBoldText = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
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
        if (maskColors.size != maskSize * maskSize) {
            return
        }
        for (i in mask.indices) {
            maskColors[i] = if (mask[i].toInt() == 1) 0x8F00E676.toInt() else 0x59FF5252.toInt()
        }
        var bmp = maskBitmap
        if (bmp == null || bmp.width != maskSize) {
            bmp = Bitmap.createBitmap(maskSize, maskSize, Bitmap.Config.ARGB_8888)
            maskBitmap = bmp
        }
        bmp.setPixels(maskColors, 0, maskSize, 0, 0, maskSize, maskSize)

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
        val p = Paint().apply { isFilterBitmap = true }
        canvas.drawBitmap(bmp, null, rect, p)
        edgePaint.strokeWidth = 3f
        edgePaint.style = Paint.Style.STROKE
        canvas.drawRect(rect, edgePaint)

        legendPaint.color = 0xFF00E676.toInt()
        canvas.drawText("绿=可通行", rect.left + 10f, rect.top + 46f, legendPaint)
        legendPaint.color = 0xFFFF5252.toInt()
        canvas.drawText("红=不可通行", rect.left + 10f, rect.top + 92f, legendPaint)
    }
}
