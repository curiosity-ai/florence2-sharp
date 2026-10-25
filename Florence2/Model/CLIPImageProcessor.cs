using System;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace Florence2;

public class CLIPImageProcessor
{

    private CLIPConfig _config;


    public CLIPImageProcessor(CLIPConfig config)
    {
        _config = config;
    }

    public (DenseTensor<float> pixel_values, (int imgWidth, int imgHeight)[] imgSizes) Preprocess(params Stream[] imgStream)
    {
        //TODO pius: this is not 100% the same as the python pillow library. The handling of JPG color profiles seems to be different.

        DenseTensor<float> input_normalized = new DenseTensor<float>(new[] { imgStream.Length, 3, _config.CropWidth, _config.CropHeight });

        (int imgWidth, int imgHeight)[] imgSizes = new (int imgWidth, int imgHeight)[imgStream.Length];

        for (int i = 0; i < imgStream.Length; i++)
        {
            using var image = LoadRgba(imgStream[i]);

            int imgHeight = image.Height;
            int imgWidth  = image.Width;

            var resized = BicubicResample.Resize(image.GetPixelSpan(), imgWidth, imgHeight, _config.CropWidth, _config.CropHeight);

            for (int y = 0; y < _config.CropHeight; y++)
            {
                for (int x = 0; x < _config.CropWidth; x++)
                {
                    int p = (y * _config.CropWidth + x) * 4;

                    input_normalized[i, 0, y, x] = ((resized[p + 2] * _config.RescaleFactor) - _config.ImageMean[0]) / _config.ImageStd[0];
                    input_normalized[i, 1, y, x] = ((resized[p + 1] * _config.RescaleFactor) - _config.ImageMean[1]) / _config.ImageStd[1];
                    input_normalized[i, 2, y, x] = ((resized[p + 0] * _config.RescaleFactor) - _config.ImageMean[2]) / _config.ImageStd[2];
                }
            }

            imgSizes[i] = (imgWidth, imgHeight);
        }


        return (input_normalized, imgSizes);
    }

    private static SKBitmap LoadRgba(Stream stream)
    {
        using var codec = SKCodec.Create(stream) ?? throw new InvalidOperationException("Unsupported or invalid image format.");

        var info   = new SKImageInfo(codec.Info.Width, codec.Info.Height, SKColorType.Rgba8888, SKAlphaType.Unpremul);
        var bitmap = SKBitmap.Decode(codec, info) ?? throw new InvalidOperationException("Failed to decode image.");
        return bitmap;
    }


    public class CLIPConfig
    {
        public float[] ImageMean      { get; set; }
        public long    ImageSeqLength { get; set; }
        public float[] ImageStd       { get; set; }
        public float   RescaleFactor  { get; set; }
        public int     CropHeight     { get; set; }
        public int     CropWidth      { get; set; }
    }


}
