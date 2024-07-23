namespace Florence2;

using System;

public class CoordinatesQuantizer
{



    private QuantizerMode _mode;
    private (int, int)    _bins;

    public CoordinatesQuantizer(QuantizerMode mode, (int, int) bins)
    {
        this._mode = mode;
        this._bins = bins;
    }

    public Coordinates<int>[] Quantize(Coordinates<float>[] coordinates, (float, float) size)
    {
        var (bins_w, bins_h) = _bins;
        var (size_w, size_h) = size;
        float size_per_bin_w = size_w / bins_w;
        float size_per_bin_h = size_h / bins_h;


        var quantizedCoordinates = new Coordinates<int>[coordinates.Length];

        for (int i = 0; i < coordinates.Length; i++)
        {
            float x = coordinates[i].x;
            float y = coordinates[i].y;

            switch (_mode)
            {

                case QuantizerMode.Floor:
                {
                    int quantized_x = (int)Math.Clamp((float)Math.Floor(x / size_per_bin_w), 0, bins_w - 1);
                    int quantized_y = (int)Math.Clamp((float)Math.Floor(y / size_per_bin_h), 0, bins_h - 1);
                    quantizedCoordinates[i] = new Coordinates<int>(quantized_x, quantized_y);
                    break;
                }
                default: throw new ArgumentOutOfRangeException();
            }
        }

        return quantizedCoordinates;
    }

    public Coordinates<float>[] Dequantize(Coordinates<int>[] coordinates, (float, float) size)
    {
        var (bins_w, bins_h) = _bins;
        var (size_w, size_h) = size;
        var size_per_bin_w = size_w / bins_w;
        var size_per_bin_h = size_h / bins_h;

        var dequantizedCoordinates = new Coordinates<float>[coordinates.Length];

        for (var i = 0; i < coordinates.Length; i++)
        {
            float x = coordinates[i].x;
            float y = coordinates[i].y;

            switch (_mode)
            {
                case QuantizerMode.Floor:
                {
                    var dequantized_x = (x + 0.5f) * size_per_bin_w;
                    var dequantized_y = (y + 0.5f) * size_per_bin_h;
                    dequantizedCoordinates[i] = new Coordinates<float>(dequantized_x, dequantized_y);
                    break;
                }
                default: throw new ArgumentOutOfRangeException();
            }
        }

        return dequantizedCoordinates;
    }
}