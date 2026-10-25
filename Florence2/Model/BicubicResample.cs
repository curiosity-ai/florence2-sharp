using System;

namespace Florence2;

/// <summary>
/// Separable bicubic resampling (a = -0.5) on RGBA8888 pixels, following the approach used by Pillow:
/// the kernel support is widened when downscaling so the result is antialiased.
/// SkiaSharp's built-in cubic sampling does not prefilter when downscaling, which causes aliasing.
/// </summary>
internal static class BicubicResample
{
    private const double Support = 2.0;

    private static double Cubic(double x)
    {
        const double a = -0.5;
        x = Math.Abs(x);
        if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
        if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
        return 0.0;
    }

    private static (int[] start, int[] count, float[][] weights) ComputeCoefficients(int inSize, int outSize)
    {
        double scale       = (double)inSize / outSize;
        double filterScale = Math.Max(scale, 1.0);
        double support     = Support * filterScale;

        var start   = new int[outSize];
        var count   = new int[outSize];
        var weights = new float[outSize][];

        for (int i = 0; i < outSize; i++)
        {
            double center = (i + 0.5) * scale;
            int    xmin   = Math.Max((int)(center - support + 0.5), 0);
            int    xmax   = Math.Min((int)(center + support + 0.5), inSize);

            var    w     = new float[xmax - xmin];
            double total = 0;

            for (int x = 0; x < w.Length; x++)
            {
                double v = Cubic((x + xmin - center + 0.5) / filterScale);
                w[x]  =  (float)v;
                total += v;
            }

            if (total != 0)
            {
                for (int x = 0; x < w.Length; x++) w[x] = (float)(w[x] / total);
            }

            start[i]   = xmin;
            count[i]   = w.Length;
            weights[i] = w;
        }

        return (start, count, weights);
    }

    public static byte[] Resize(ReadOnlySpan<byte> src, int srcWidth, int srcHeight, int dstWidth, int dstHeight)
    {
        // Horizontal pass: srcWidth x srcHeight -> dstWidth x srcHeight
        var (hStart, hCount, hWeights) = ComputeCoefficients(srcWidth, dstWidth);
        var tmp = new float[dstWidth * srcHeight * 4];

        for (int y = 0; y < srcHeight; y++)
        {
            int srcRow = y * srcWidth * 4;
            int tmpRow = y * dstWidth * 4;

            for (int x = 0; x < dstWidth; x++)
            {
                float r = 0, g = 0, b = 0, a = 0;
                var   w = hWeights[x];
                int   p = srcRow + hStart[x] * 4;

                for (int k = 0; k < hCount[x]; k++, p += 4)
                {
                    r += src[p]     * w[k];
                    g += src[p + 1] * w[k];
                    b += src[p + 2] * w[k];
                    a += src[p + 3] * w[k];
                }

                int t = tmpRow + x * 4;
                tmp[t]     = r;
                tmp[t + 1] = g;
                tmp[t + 2] = b;
                tmp[t + 3] = a;
            }
        }

        // Vertical pass: dstWidth x srcHeight -> dstWidth x dstHeight
        var (vStart, vCount, vWeights) = ComputeCoefficients(srcHeight, dstHeight);
        var dst      = new byte[dstWidth * dstHeight * 4];
        int rowWidth = dstWidth * 4;

        for (int y = 0; y < dstHeight; y++)
        {
            var w      = vWeights[y];
            int dstRow = y * rowWidth;

            for (int c = 0; c < rowWidth; c++)
            {
                float v = 0;
                int   p = vStart[y] * rowWidth + c;

                for (int k = 0; k < vCount[y]; k++, p += rowWidth)
                {
                    v += tmp[p] * w[k];
                }

                dst[dstRow + c] = (byte)Math.Clamp((int)MathF.Round(v), 0, 255);
            }
        }

        return dst;
    }
}
