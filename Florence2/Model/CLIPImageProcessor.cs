using System;
using System.IO;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Processing.Processors.Transforms;

namespace Florence2;

public class CLIPImageProcessor
{

    private CLIPConfig _config;
    private IResampler _resampler;


    public CLIPImageProcessor(CLIPConfig config, IResampler? resampler = null)
    {
        _config    = config;
        _resampler = resampler ?? KnownResamplers.Bicubic;

    }

    public (DenseTensor<float> pixel_values, (int imgWidth, int imgHeight) imgSize) Preprocess(Stream imgStream)
    {
        //TODO pius: this is not 100% the same as the python pillow library. The handling of JPG color profiles seems to be different as well as how the resizing alhorithm works. (even though both ar Bicubic)

        DenseTensor<float> input_normalized = new DenseTensor<float>(new[] { 3, _config.CropWidth, _config.CropHeight });

        int imgHeight = -1;
        int imgWidth  = -1;

        using (var image = Image.Load<Rgba32>(imgStream))
        {
            imgHeight = image.Height;
            imgWidth  = image.Width;

            image.Mutate(x => x.Resize(_config.CropWidth, _config.CropHeight, _resampler, false));

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Rgba32> pixelSpan = accessor.GetRowSpan(y);

                    for (int x = 0; x < accessor.Width; x++)
                    {
                        input_normalized[0, y, x] = ((pixelSpan[x].B * _config.RescaleFactor) - _config.ImageMean[0]) / _config.ImageStd[0];
                        input_normalized[1, y, x] = ((pixelSpan[x].G * _config.RescaleFactor) - _config.ImageMean[1]) / _config.ImageStd[1];
                        input_normalized[2, y, x] = ((pixelSpan[x].R * _config.RescaleFactor) - _config.ImageMean[2]) / _config.ImageStd[2];
                    }
                }
            });
        }

        return (input_normalized, (imgWidth, imgHeight));
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