using System.Diagnostics;
using System.Numerics;
using System.Text;
using System.Text.Json;
using Florence2;
using Microsoft.Extensions.Logging;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.Drawing.Processing;
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

        foreach (var task in Enum.GetValues<TaskTypes>())
        {
            using var imgStream       = LoadImage("book.jpg");
            using var imgStreamResult = LoadImage("book.jpg");

            var results = modelSession.Run(task, imgStream, textInput: "DUANE", CancellationToken.None);

            DrawInline(imgStreamResult, task, "DUANE", results, outFolder: outPath);

            logger?.ZLogInformation($"{task} : {JsonSerializer.Serialize(results)}");
        }

        foreach (var task in Enum.GetValues<TaskTypes>())
        {
            using var imgStream       = LoadImage("car.jpg");
            using var imgStreamResult = LoadImage("car.jpg");

            var results = modelSession.Run(task, imgStream, textInput: "window", CancellationToken.None);
            DrawInline(imgStreamResult, task, "window", results, outFolder: outPath);

            logger?.ZLogInformation($"{task} : {JsonSerializer.Serialize(results)}");
        }
    }

    private static Stream LoadImage(string path) => File.OpenRead(path);

    private static void DrawInline(Stream imgStreamResult, TaskTypes task, string userText, FlorenceResults[] results, string? outFolder = null)
    {
        if (!results.Any(r => (r.OCRBBox is object && r.OCRBBox.Any())
         || (r.BoundingBoxes is object             && r.BoundingBoxes.Any())
         || (r.Polygons is object                  && r.Polygons.Any()))) return;

        outFolder ??= Environment.GetFolderPath(Environment.SpecialFolder.Desktop);

        var penBox = Pens.Solid(Color.Red, 1.0f);

        if (Florence2Model.TaskPromptsWithoutInputsDict.ContainsKey(task))
        {
            userText = "";
        }

        var fontFamily = DefaultFont.Value;
        var font = fontFamily.CreateFont(12, FontStyle.Italic);

        using (var image = Image.Load<Rgba32>(imgStreamResult))
        {
            image.Mutate(x =>
            {
                for (var index = 0; index < results.Length; index++)
                {
                    var finalResult = results[index];

                    if (finalResult.BoundingBoxes is object)
                    {
                        var i = 0;

                        foreach (var bbox1 in finalResult.BoundingBoxes)
                        {
                            PointF? labelPoint = null;

                            foreach (var bboxBBox in bbox1.BBoxes)
                            {
                                var polygon = new List<PointF>();
                                var p       = new PointF(bboxBBox.xmin, bboxBBox.ymin);

                                labelPoint ??= p;

                                polygon.Add(p);
                                polygon.Add(new PointF(bboxBBox.xmin, bboxBBox.ymax));
                                polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymax));
                                polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymin));

                                x.DrawPolygon(penBox, polygon.ToArray());

                            }

                            var label = bbox1.Label;
                            x.DrawText(label, font, Brushes.Solid(Color.Black), Pens.Solid(Color.White, 1), labelPoint.Value);
                            i++;
                        }

                    }

                    if (finalResult.OCRBBox is object)
                    {
                        foreach (var labledOcr in finalResult.OCRBBox)
                        {
                            var polygon = labledOcr.QuadBox.Select(e => new PointF(e.x, e.y)).ToArray();
                            x.DrawPolygon(penBox, polygon);
                            var textZero = polygon.First();
                            x.DrawText(labledOcr.Text, font, Brushes.Solid(Color.Black), Pens.Solid(Color.White, 1), textZero);

                        }
                    }

                    if (finalResult.Polygons is object)
                    {
                        foreach (var finalResultPolygon in finalResult.Polygons)
                        {
                            PointF? labelPoint = null;

                            if (finalResultPolygon.Polygon is object)
                            {
                                var polygon1 = finalResultPolygon.Polygon.Select(e =>
                                {
                                    var p = new PointF(e.x, e.y);
                                    labelPoint ??= p;
                                    return p;
                                }).ToArray();
                                x.DrawPolygon(penBox, polygon1);
                            }

                            if (finalResultPolygon.BBoxes is object)
                            {
                                foreach (var bboxBBox in finalResultPolygon.BBoxes)
                                {
                                    var polygon = new List<PointF>();
                                    var p       = new PointF(bboxBBox.xmin, bboxBBox.ymin);

                                    labelPoint ??= p;

                                    polygon.Add(p);
                                    polygon.Add(new PointF(bboxBBox.xmin, bboxBBox.ymax));
                                    polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymax));
                                    polygon.Add(new PointF(bboxBBox.xmax, bboxBBox.ymin));

                                    x.DrawPolygon(penBox, polygon.ToArray());

                                }
                            }

                            x.DrawText(finalResultPolygon.Label, font, Brushes.Solid(Color.Black), Pens.Solid(Color.White, 1), labelPoint.Value);
                        }
                    }

                }
            });

            image.SaveAsBmp($"{outFolder}/book-{task}-{userText}.bmp");
        }
    }

    private static Lazy<FontFamily> DefaultFont = new Lazy<FontFamily>(() => GetDefaultFont());
    private static FontFamily GetDefaultFont()
    {
        FontFamily? best = null;

        if (OperatingSystem.IsWindows() || OperatingSystem.IsMacOS())
        {
            best = SystemFonts.Get("Arial");
        }
        else if (OperatingSystem.IsLinux())
        {
            best = SystemFonts.TryGet("Arial", out var arial) ? arial :
                SystemFonts.TryGet("Ubuntu", out var sf) ? sf :
                SystemFonts.TryGet("Liberation Sans", out var ls) ? ls :
                SystemFonts.TryGet("DejaVu Sans", out var dvs) ? dvs :
                SystemFonts.TryGet("Rasa", out var rasa) ? rasa :
                SystemFonts.TryGet("FreeSans", out var fs) ? fs :
                                                                    null;
        }
        return best ?? SystemFonts.Families.FirstOrDefault(f => f.Name.Contains("Sans"), SystemFonts.Families.First());
    }
}