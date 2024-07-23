using System;

namespace Florence2;

public enum QuantizerMode
{
    Floor
}

public class BoxQuantizer
{
    private QuantizerMode _mode;
    private (int, int)    _bins;

    public BoxQuantizer(QuantizerMode mode, (int, int) bins)
    {
        _mode = mode;
        _bins = bins;
    }

    public BoundingBox<int>[] Quantize(BoundingBox<float>[] boxes, (int, int) size)
    {
        var (bins_w, bins_h) = _bins;
        var (size_w, size_h) = size;
        var size_per_bin_w = (float)size_w / bins_w;
        var size_per_bin_h = (float)size_h / bins_h;

        var quantized_boxes = new BoundingBox<int>[boxes.Length];

        for (var i = 0; i < boxes.Length; i++)
        {
            switch (_mode)
            {
                case QuantizerMode.Floor:
                {
                    quantized_boxes[i] = new BoundingBox<int>(
                        xmin: Math.Clamp((int)Math.Floor(boxes[i].xmin / size_per_bin_w), 0, bins_w - 1),
                        ymin: Math.Clamp((int)Math.Floor(boxes[i].ymin / size_per_bin_h), 0, bins_h - 1),
                        xmax: Math.Clamp((int)Math.Floor(boxes[i].xmax / size_per_bin_w), 0, bins_w - 1),
                        ymax: Math.Clamp((int)Math.Floor(boxes[i].ymax / size_per_bin_h), 0, bins_h - 1));
                    break;
                }
                default: throw new ArgumentException("Incorrect quantization type.");

            }
        }

        return quantized_boxes;
    }

    public BoundingBox<float>[] Dequantize(BoundingBox<int>[] boxes, (int, int) size)
    {
        var (bins_w, bins_h) = _bins;
        var (size_w, size_h) = size;
        var size_per_bin_w = (float)size_w / bins_w;
        var size_per_bin_h = (float)size_h / bins_h;

        var dequantized_boxes = new BoundingBox<float>[boxes.Length];

        for (var i = 0; i < boxes.Length; i++)
        {
            switch (_mode)
            {
                case QuantizerMode.Floor:
                {
                    dequantized_boxes[i] = new BoundingBox<float>(
                        xmin: (boxes[i].xmin + 0.5f) * size_per_bin_w,
                        ymin: (boxes[i].ymin + 0.5f) * size_per_bin_h,
                        xmax: (boxes[i].xmax + 0.5f) * size_per_bin_w,
                        ymax: (boxes[i].ymax + 0.5f) * size_per_bin_h);
                    break;
                }
                default: throw new ArgumentOutOfRangeException();
            }
        }

        return dequantized_boxes;
    }
}