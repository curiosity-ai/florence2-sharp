using System.Diagnostics;
using System.Net.Http.Headers;
using ImageToTextTransformer;
using Microsoft.Extensions.Logging;
using ZLogger;

namespace Test;

public class FlorenceModelDownloader : IModelSource
{
    private static readonly Mutex _florence2Mutex = new Mutex(false, @"Global\image-to-text-transformers-florence2-install"); //Need to use a named mutex as this can happen across processes

    private Dictionary<IModelSource.Model, string> _modelPaths = new Dictionary<IModelSource.Model, string>();

    private static string GetModelFileName(IModelSource.Model model) => model switch
    {
        IModelSource.Model.DecoderModelMerged => "decoder_model_merged.onnx",
        IModelSource.Model.EmbedTokens        => "embed_tokens.onnx",
        IModelSource.Model.EncoderModel       => "encoder_model.onnx",
        IModelSource.Model.VisionEncoder      => "vision_encoder.onnx",
        _                                     => throw new ArgumentOutOfRangeException(nameof(model), model, null)
    };

    private string _modelFolderBasePath;
    public FlorenceModelDownloader(string modelFolderBasePath)
    {
        _modelFolderBasePath = modelFolderBasePath;
        Directory.CreateDirectory(_modelFolderBasePath);
    }

    private Task DownloadModel(IModelSource.Model model, Action<DownloadStatus> onStatusUpdate, ILogger logger = null, CancellationToken ct = default)
    {
        //TODO add lock for multiple call
        var modelFileName = GetModelFileName(model);
        var filePath      = Path.Combine(_modelFolderBasePath, modelFileName);

        if (!File.Exists(filePath) || new FileInfo(filePath).Length == 0)
        {
            //Download in a separate thread because the Mutex is thread-affine

            var tcs = new TaskCompletionSource<string>();

            var downloadThread = new Thread(() =>
            {
                if (_florence2Mutex.WaitOne())
                {
                    try
                    {
                        if (!File.Exists(filePath) || new FileInfo(filePath).Length == 0)
                        {
                            onStatusUpdate(new DownloadStatus
                            {
                                Message  = $"Downloading Florence2 Model {model}",
                                Progress = 0f
                            });

                            logger?.ZLogInformation("Downloading Florence2 Model {0} to {1}", model, filePath);

                            var requestUri = $"https://models.curiosity.ai/florence-2/{modelFileName}";


                            var request = new System.Net.Http.HttpRequestMessage(HttpMethod.Get, requestUri);

                            using var httpClient = new HttpClient();

                            var sw = Stopwatch.StartNew();

                            var response = httpClient.Send(request, HttpCompletionOption.ResponseHeadersRead, ct);

                            if (response.IsSuccessStatusCode)
                            {
                                var totalDownloadSize = response.Content?.Headers?.ContentLength ?? -1;

                                onStatusUpdate(GetStatus(totalDownloadSize, 0, 0, TimeSpan.Zero, $"Downloading Florence2 Model {model}"));


                                var totalBytesRead    = 0L;
                                var previousBytesRead = 0L;
                                var buffer            = new byte[2 << 18]; // 512kb
                                var isMoreToRead      = true;


                                using (var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 2 << 18 /* 512kb */, true))
                                {
                                    bool finished = false;

                                    while (!finished)
                                    {
                                        try
                                        {
                                            using (var contentStream = response.Content.ReadAsStream(ct))
                                            {
                                                do
                                                {
                                                    var bytesRead = contentStream.Read(buffer, 0, buffer.Length);

                                                    if (bytesRead == 0)
                                                    {
                                                        finished     = true;
                                                        isMoreToRead = false;
                                                        onStatusUpdate(GetStatus(totalDownloadSize, totalBytesRead, totalBytesRead, TimeSpan.Zero, $"Downloading Florence2 Model {model}"));

                                                        continue;
                                                    }

                                                    fileStream.Write(buffer, 0, bytesRead);

                                                    totalBytesRead += bytesRead;

                                                    var elapsed = sw.Elapsed;

                                                    if (elapsed.TotalSeconds > 1)
                                                    {
                                                        onStatusUpdate(GetStatus(totalDownloadSize, totalBytesRead, previousBytesRead, elapsed, $"Downloading Florence2 Model {model}"));
                                                        sw                = Stopwatch.StartNew();
                                                        previousBytesRead = totalBytesRead;
                                                    }

                                                } while (isMoreToRead);
                                            }
                                        }
                                        catch (Exception exception)
                                        {
                                            logger?.ZLogWarning(exception, "Error downloading model from {0}, will try again", requestUri);
                                            Thread.Sleep(5000);

                                            var newRequest = new HttpRequestMessage(HttpMethod.Get, requestUri);

                                            if (response.Headers.AcceptRanges.Contains("bytes"))
                                            {
                                                newRequest.Headers.Range = new RangeHeaderValue(fileStream.Position, null);
                                            }
                                            else
                                            {
                                                totalBytesRead      = 0;
                                                previousBytesRead   = 0;
                                                fileStream.Position = 0;
                                            }

                                            response = httpClient.Send(newRequest, HttpCompletionOption.ResponseHeadersRead, ct);
                                        }
                                    }
                                }

                                logger?.ZLogInformation("Downloaded Florence2 Model {0} to {1}", model, filePath);

                                onStatusUpdate(GetStatus(totalDownloadSize, totalDownloadSize, 0, TimeSpan.Zero, $"Downloading Florence2 Model {model}"));

                            }
                            else
                            {
                                onStatusUpdate(new DownloadStatus
                                {
                                    Message  = $"Downloading Florence2 Model {model}",
                                    Progress = 0f
                                });
                            }
                        }
                        tcs.TrySetResult(filePath);
                    }
                    catch (Exception E)
                    {
                        tcs.TrySetException(E);

                        if (File.Exists(filePath))
                        {
                            try
                            {
                                File.Delete(filePath);
                            }
                            catch (Exception IE)
                            {
                                logger?.ZLogError(IE, "Error deleting {0}", filePath);

                                onStatusUpdate(new DownloadStatus
                                {
                                    Error    = "Failed to download model: " + IE.Message,
                                    Progress = 0f
                                });
                            }
                        }
                    }
                    finally
                    {
                        _florence2Mutex.ReleaseMutex();
                    }
                }
            });

            downloadThread.IsBackground = true;
            downloadThread.Name         = "Florence2 Downloader";
            downloadThread.Start();
            return tcs.Task;
        }
        else
        {
            logger?.ZLogDebug($"model {model} already downloaded");
            _modelPaths.Add(model, filePath);
            return Task.CompletedTask;
        }
    }

    public async Task InitModelRepo(Action<IStatus> onStatusUpdate, ILogger logger = null, CancellationToken ct = default)
    {
        foreach (var model in Enum.GetValues<IModelSource.Model>())
        {
            if (!_modelPaths.TryGetValue(model, out var modelPath))
            {
                await DownloadModel(model, onStatusUpdate, logger, ct);
            }
        }
    }

    public bool TryGetModelPath(IModelSource.Model model, out string modelPath)
    {
        if (_modelPaths.TryGetValue(model, out modelPath))
        {
            return true;
        }
        else
        {
            throw new Exception(nameof(FlorenceModelDownloader) + " was not initialized for " + model + ". Call InitModelRepo first.");
        }
    }

    public byte[] GetModelBytes(IModelSource.Model model)
    {
        if (_modelPaths.TryGetValue(model, out var modelPath))
        {
            using var fs = File.OpenRead(modelPath);
            var       b  = new byte[fs.Length];
            var       ms = new MemoryStream(b);
            fs.CopyTo(ms);
            return b;
        }
        else
        {
            throw new Exception(nameof(FlorenceModelDownloader) + " was not initialized for " + model + ". Call InitModelRepo first.");
        }
    }



    private static DownloadStatus GetStatus(long totalDownloadSize, long totalBytesRead, long previousBytesRead, TimeSpan elapsed, string message = null)
    {
        var ts = elapsed.TotalSeconds;

        if (ts > 0)
        {
            double speed = ((double)(totalBytesRead - previousBytesRead)) / ts;

            return new DownloadStatus
            {
                Progress = (float)((double)totalBytesRead / (double)totalDownloadSize),
                Message  = $"{ByteFormatter.FormatBytes(totalBytesRead)} / {ByteFormatter.FormatBytes(totalDownloadSize)} ({ByteFormatter.FormatBytes(speed)}/s) " + message
            };
        }
        else
        {
            return new DownloadStatus
            {
                Progress = (float)((double)totalBytesRead / (double)totalDownloadSize),
                Message  = $"{ByteFormatter.FormatBytes(totalBytesRead)} / {ByteFormatter.FormatBytes(totalDownloadSize)} " + message
            };
        }
    }

    public class DownloadStatus : IStatus
    {
        public string Error    { get; set; }
        public string Message  { get; set; }
        public float  Progress { get; set; }
    }

}
public interface IStatus
{
    public string Error    { get; set; }
    public string Message  { get; set; }
    public float  Progress { get; set; }
}