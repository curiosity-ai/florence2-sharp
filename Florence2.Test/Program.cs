using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Text.Json;
using Florence2;
using Microsoft.Extensions.Logging;
using SkiaSharp;
using ZLogger;

namespace Florence2.Test;

public static class Programm
{
    static async Task Main(string[] args)
    {

        using ILoggerFactory factory = LoggerFactory.Create(builder => builder.AddZLoggerConsole());
        ILogger              logger  = factory.CreateLogger("Florence-2 Test");

        var modelSource = new FlorenceModelDownloader("./models");

        var outPath = "./output";

        if (Directory.Exists(outPath)) Directory.Delete(outPath, true);

        Directory.CreateDirectory(outPath);

        await modelSource.DownloadModelsAsync(status => logger?.ZLogInformation($"{status.Progress:P0} {status.Error} {status.Message}"), logger, CancellationToken.None);

        var modelSession = new Florence2Model(modelSource);

        foreach (var task in new TaskTypes[] { TaskTypes.OCR_WITH_REGION })
        {
            var sw = Stopwatch.StartNew();
//            using var imgStream2 = LoadImage("ocrsample.png");
            using var imgStream  = LoadImage("book.jpg");
//            using var imgStreamResult = LoadImage("ocrsample.png");

            var results = modelSession.Run(task, new[] {  imgStream }, textInput: null, CancellationToken.None);
            
//            DrawInline(imgStreamResult, task, userText: null, results, outFolder: outPath);

            logger?.ZLogInformation($"{task} : {JsonSerializer.Serialize(results, new JsonSerializerOptions() { WriteIndented = true })}");
            logger.ZLogInformation($"Total: {sw.Elapsed:g}");

        }
    }

    private static Stream LoadImage(string path) => File.OpenRead(path);

    private static void DrawInline(Stream imgStreamResult, TaskTypes task, string userText, FlorenceResults[] results, string? outFolder = null)
    {
        if (!results.Any(r => (r.OCRBBox is object && r.OCRBBox.Any())
         || (r.BoundingBoxes is object             && r.BoundingBoxes.Any())
         || (r.Polygons is object                  && r.Polygons.Any()))) return;

        outFolder ??= Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

        using var penBox = new SKPaint { Color = SKColors.Red, Style = SKPaintStyle.Stroke, StrokeWidth = 1.0f, IsAntialias = true };
        using var textFill = new SKPaint { Color = SKColors.Black, Style = SKPaintStyle.Fill, IsAntialias = true };
        using var textStroke = new SKPaint { Color = SKColors.White, Style = SKPaintStyle.Stroke, StrokeWidth = 1.0f, IsAntialias = true };

        if (Florence2Model.TaskPromptsWithoutInputsDict.ContainsKey(task))
        {
            userText = "";
        }

        using var font = new SKFont(DefaultTypeface.Value, 12);

        void DrawPolygon(SKCanvas canvas, SKPoint[] points)
        {
            canvas.DrawPoints(SKPointMode.Polygon, points.Append(points[0]).ToArray(), penBox);
        }

        void DrawText(SKCanvas canvas, string text, SKPoint point)
        {
            // Draw from the top-left corner like the polygon origin, so offset by the font ascent
            var baseline = new SKPoint(point.X, point.Y - font.Metrics.Ascent);
            canvas.DrawText(text, baseline, SKTextAlign.Left, font, textStroke);
            canvas.DrawText(text, baseline, SKTextAlign.Left, font, textFill);
        }

        SKPoint[] BoxToPolygon((float xmin, float ymin, float xmax, float ymax) box) => new[]
        {
            new SKPoint(box.xmin, box.ymin),
            new SKPoint(box.xmin, box.ymax),
            new SKPoint(box.xmax, box.ymax),
            new SKPoint(box.xmax, box.ymin),
        };

        using (var image = SKBitmap.Decode(imgStreamResult))
        using (var canvas = new SKCanvas(image))
        {
            for (var index = 0; index < results.Length; index++)
            {
                var finalResult = results[index];

                if (finalResult.BoundingBoxes is object)
                {
                    foreach (var bbox1 in finalResult.BoundingBoxes)
                    {
                        SKPoint? labelPoint = null;

                        foreach (var bboxBBox in bbox1.BBoxes)
                        {
                            var polygon = BoxToPolygon((bboxBBox.xmin, bboxBBox.ymin, bboxBBox.xmax, bboxBBox.ymax));
                            labelPoint ??= polygon[0];
                            DrawPolygon(canvas, polygon);
                        }

                        DrawText(canvas, bbox1.Label, labelPoint.Value);
                    }
                }

                if (finalResult.OCRBBox is object)
                {
                    foreach (var labledOcr in finalResult.OCRBBox)
                    {
                        var polygon = labledOcr.QuadBox.Select(e => new SKPoint(e.x, e.y)).ToArray();
                        DrawPolygon(canvas, polygon);
                        DrawText(canvas, labledOcr.Text, polygon.First());
                    }
                }

                if (finalResult.Polygons is object)
                {
                    foreach (var finalResultPolygon in finalResult.Polygons)
                    {
                        SKPoint? labelPoint = null;

                        if (finalResultPolygon.Polygon is object)
                        {
                            var polygon1 = finalResultPolygon.Polygon.Select(e => new SKPoint(e.x, e.y)).ToArray();
                            labelPoint ??= polygon1.FirstOrDefault();
                            DrawPolygon(canvas, polygon1);
                        }

                        if (finalResultPolygon.BBoxes is object)
                        {
                            foreach (var bboxBBox in finalResultPolygon.BBoxes)
                            {
                                var polygon = BoxToPolygon((bboxBBox.xmin, bboxBBox.ymin, bboxBBox.xmax, bboxBBox.ymax));
                                labelPoint ??= polygon[0];
                                DrawPolygon(canvas, polygon);
                            }
                        }

                        DrawText(canvas, finalResultPolygon.Label, labelPoint.Value);
                    }
                }
            }

            canvas.Flush();

            using var data = image.Encode(SKEncodedImageFormat.Png, 100);
            using var file = File.Create($"{outFolder}/book-{task}-{userText}.png");
            data.SaveTo(file);
        }
    }

    private static Lazy<SKTypeface> DefaultTypeface = new Lazy<SKTypeface>(() => GetDefaultTypeface());
    private static SKTypeface GetDefaultTypeface()
    {
        var candidates = OperatingSystem.IsLinux()
            ? new[] { "Arial", "Ubuntu", "Liberation Sans", "DejaVu Sans", "Rasa", "FreeSans" }
            : new[] { "Arial" };

        var style = new SKFontStyle(SKFontStyleWeight.Normal, SKFontStyleWidth.Normal, SKFontStyleSlant.Italic);

        foreach (var name in candidates)
        {
            var typeface = SKFontManager.Default.MatchFamily(name, style);
            if (typeface is object) return typeface;
        }

        if (OperatingSystem.IsLinux())
        {
            // Builds of Skia without fontconfig can't resolve family names, so look for a font file directly
            var fontFiles = new[] { "/usr/share/fonts", "/usr/local/share/fonts" }
               .Where(Directory.Exists)
               .SelectMany(d => Directory.EnumerateFiles(d, "*.ttf", SearchOption.AllDirectories))
               .ToArray();

            foreach (var name in candidates.Select(c => c.Replace(" ", "")))
            {
                var file = fontFiles.FirstOrDefault(f => Path.GetFileNameWithoutExtension(f).StartsWith(name, StringComparison.OrdinalIgnoreCase));
                if (file is object && SKTypeface.FromFile(file) is SKTypeface typeface) return typeface;
            }
        }

        return SKTypeface.FromFamilyName(null, style) ?? SKTypeface.Default;
    }
}
