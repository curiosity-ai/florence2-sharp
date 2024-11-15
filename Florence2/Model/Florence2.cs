using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.Processing;

namespace Florence2;

public interface IModelSource
{
    public enum Model
    {
        DecoderModelMerged,
        EmbedTokens,
        EncoderModel,
        VisionEncoder
    }
    public bool TryGetModelPath(IModelSource.Model model, out string modelPath);

    public byte[] GetModelBytes(IModelSource.Model model);

}
public class Florence2Model
{
    private readonly  SessionOptions         _sessionOptions;
    private readonly  InferenceSession       _sessionDecoderMerged;
    private readonly  InferenceSession       _sessionEmbedTokens;
    private readonly  InferenceSession       _sessionEncoder;
    private readonly  InferenceSession       _sessionVisionEncoder;
    internal readonly Florence2Tokenizer     _tokenizer;
    private readonly  CLIPImageProcessor     _imageProcessor;
    private readonly  Florence2PostProcessor _postProcessor;


    private InferenceSession GetSessionForModel(IModelSource source, IModelSource.Model model)
    {
        return source.TryGetModelPath(model, out var modelPath)
            ? new InferenceSession(modelPath,                   _sessionOptions)
            : new InferenceSession(source.GetModelBytes(model), _sessionOptions);

    }

    public Florence2Model(IModelSource modelSource, SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();

        _sessionDecoderMerged = GetSessionForModel(modelSource, IModelSource.Model.DecoderModelMerged);
        _sessionEmbedTokens   = GetSessionForModel(modelSource, IModelSource.Model.EmbedTokens);
        _sessionEncoder       = GetSessionForModel(modelSource, IModelSource.Model.EncoderModel);
        _sessionVisionEncoder = GetSessionForModel(modelSource, IModelSource.Model.VisionEncoder);

        _tokenizer = Florence2Tokenizer.Init();

        _imageProcessor = new CLIPImageProcessor(new CLIPImageProcessor.CLIPConfig()
        {
            ImageMean      = [0.485f, 0.456f, 0.406f],
            ImageSeqLength = 577,
            ImageStd       = [0.229f, 0.224f, 0.225f],
            RescaleFactor  = 0.00392156862745098f,
            CropHeight     = 768,
            CropWidth      = 768,
        }, KnownResamplers.Bicubic);
        _postProcessor = new Florence2PostProcessor();

    }


    private string ConstructPrompts(TaskTypes taskType, string textInput = null)
    {
        if (TaskPromptsWithoutInputsDict.TryGetValue(taskType, out var taskPrompt))
        {
            return taskPrompt;
        }
        else if (TaskPromptsWithInputDict.TryGetValue(taskType, out var taskPromptFormat))
        {
            if (textInput is null) throw new ArgumentNullException(nameof(textInput), "expected text with this taskType");
            return string.Format(taskPromptFormat, textInput);
        }
        else
        {
            throw new ArgumentException("not found task type" + taskType, nameof(taskType));
        }
    }

    public FlorenceResults[] Run(TaskTypes task, Stream[] imgStreams, string textInput, CancellationToken cancellationToken)
    {
        using var runOptions = new RunOptions();

        var prompts = Enumerable.Repeat(ConstructPrompts(task, textInput), imgStreams.Length).ToArray();

        var (inputIdsForEncoder, attentionMaskForEncoder) = GetTextInputs(prompts);

        var (pixelValues, imgSizes) = _imageProcessor.Preprocess(imgStreams);

        using var registration = cancellationToken.Register(() => runOptions.Terminate = true);

        using var text_features = _sessionEmbedTokens.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", inputIdsForEncoder), }, new[] { "inputs_embeds" }, runOptions);
        var       inputsEmbeds  = text_features[0].AsTensor<float>().ToDenseTensor();

        using var imageFeaturesResult = _sessionVisionEncoder.Run(new[] { NamedOnnxValue.CreateFromTensor("pixel_values", pixelValues), }, new[] { "image_features" }, runOptions);
        var       imageFeatures       = imageFeaturesResult[0].AsTensor<float>().ToDenseTensor();

        var (inputsEmbedsMerged, attentionMaskMerged) = MergeInputIdsWithImageFeatures(inputsEmbeds, imageFeatures, attentionMaskForEncoder);

        using var forwardOut = _sessionEncoder.Run(new[] { NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskMerged), NamedOnnxValue.CreateFromTensor("inputs_embeds", inputsEmbedsMerged), }, new[] { "last_hidden_state" }, runOptions);

        var lastHiddenState = forwardOut[0].AsTensor<float>().ToDenseTensor();

        var encoderOutputs = lastHiddenState;

        var result = GenerationLoop(attentionMaskMerged, encoderOutputs, runOptions);

        return result.Select((r, i) => _postProcessor.PostProcessGeneration(r, task, imgSizes[i])).ToArray();

    }

    private List<string> GenerationLoop(DenseTensor<long> attentionMask, DenseTensor<float> encoder_outputs, RunOptions runOptions)
    {
        var batchSize = attentionMask.Dimensions[0];
        var maxLength = GenerationConfig.MaxLength;
        var numBeams  = GenerationConfig.NumBeams;
        var topK      = GenerationConfig.TopK;

        int noRepeatNgramSize = GenerationConfig.NoRepeatNgramSize;

        var decoderStartTokenID = _tokenizer.TokenToID(_tokenizer.Tokens.EndOfSequence);

        var          decoderInputIds = TensorExtension.OnesLong(new[] { batchSize, 1 }, decoderStartTokenID);
        List<long>[] allInputIds     = Enumerable.Range(0, batchSize).Select(_ => new List<long>(new[] { (long)decoderStartTokenID })).ToArray();

        var results = new List<string>();

        NamedOnnxValue[] pastKeyValues = null;

        var logitsProcessors = new List<LogitsProcessor>();

        logitsProcessors.Add(new NoRepeatNGramLogitsProcessor(noRepeatNgramSize));
        logitsProcessors.Add(new ForcedBOSTokenLogitsProcessor(_tokenizer.TokenToID(_tokenizer.Tokens.BeginningOfSequence)));
        logitsProcessors.Add(new ForcedEOSTokenLogitsProcessor(maxLength, _tokenizer.TokenToID(_tokenizer.Tokens.EndOfSequence)));

        var sampler = new BeamSearchSampler(TensorOperationRegistry.TopKSession(_sessionOptions), topK: topK, numBeams: numBeams);

        var eosToken = _tokenizer.TokenToID(_tokenizer.Tokens.EndOfSequence);

        var stoppingCriteria = new List<StoppingCriteria>();
        stoppingCriteria.Add(new MaxLengthCriteria(maxLength));
        stoppingCriteria.Add(new EosTokenCriteria(eosToken));

        var decoder = new ByteLevelDecoder(_tokenizer.AddedTokens);


        double[] scores = new double[batchSize];
        var      isDone = new bool[batchSize];

        while (true)
        {
            using var decoderInputsEmbeds = _sessionEmbedTokens.Run(new[] { NamedOnnxValue.CreateFromTensor("input_ids", decoderInputIds), }, new[] { "inputs_embeds" }, runOptions); // inputIds -> input_embeds


            var useCacheBranche = pastKeyValues is object;
            var useCacheBranch  = new DenseTensor<bool>(new[] { useCacheBranche }, dimensions: new[] { 1 });

            var decoderInputsEmbedsVec = decoderInputsEmbeds[0].AsTensor<float>().ToDenseTensor();

            var decoderFeeds = new[]
            {
                NamedOnnxValue.CreateFromTensor("inputs_embeds",          decoderInputsEmbedsVec),
                NamedOnnxValue.CreateFromTensor("encoder_attention_mask", attentionMask),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states",  encoder_outputs),
                NamedOnnxValue.CreateFromTensor("use_cache_branch",       useCacheBranch),
            };

            pastKeyValues ??= InitPastKeyValues(new NormalizedConfig()).ToArray();

            if (pastKeyValues is object)
            {
                decoderFeeds = decoderFeeds.Concat(pastKeyValues).ToArray();
            }
            using var decoder_out = _sessionDecoderMerged.Run(decoderFeeds, new[] { "logits", "present.0.decoder.key", "present.0.decoder.value", "present.0.encoder.key", "present.0.encoder.value", "present.1.decoder.key", "present.1.decoder.value", "present.1.encoder.key", "present.1.encoder.value", "present.2.decoder.key", "present.2.decoder.value", "present.2.encoder.key", "present.2.encoder.value", "present.3.decoder.key", "present.3.decoder.value", "present.3.encoder.key", "present.3.encoder.value", "present.4.decoder.key", "present.4.decoder.value", "present.4.encoder.key", "present.4.encoder.value", "present.5.decoder.key", "present.5.decoder.value", "present.5.encoder.key", "present.5.encoder.value" }, runOptions);

            pastKeyValues = FromPresent(decoder_out, useCacheBranche, pastKeyValues).ToArray();

            var logits = decoder_out.First(t => t.Name == "logits");

            var logitsTensor = logits.AsTensor<float>().ToDenseTensor();

            var logitsTensorProcessed = new DenseTensor<float>(new Memory<float>(logitsTensor.ToArray()), new[] { logitsTensor.Dimensions[0], logitsTensor.Dimensions[2] }, logitsTensor.IsReversedStride);

            var generatedInputIds = new long[batchSize];

            for (int batchIndex = 0; batchIndex < batchSize; batchIndex++)
            {
                if (isDone[batchIndex])
                {
                    generatedInputIds[batchIndex] = eosToken;
                    continue;
                }

                foreach (var logitsProcessor in logitsProcessors)
                {
                    logitsProcessor.Process(batchIndex, allInputIds[batchIndex].ToArray(), logitsTensorProcessed);
                }

                var sampledTokens = sampler.Sample(batchIndex, logitsTensorProcessed);

                foreach (var (token, score) in sampledTokens)
                {
                    scores[batchIndex] += score;
                    var batchAllInputIds = allInputIds[batchIndex] ?? new List<long>();
                    batchAllInputIds.Add(token);
                    allInputIds[batchIndex] = batchAllInputIds;

                    generatedInputIds[batchIndex] = token;
                    break;
                }
            }


            foreach (var stoppingCriterion in stoppingCriteria)
            {
                var criterionDone = stoppingCriterion.Call(allInputIds, scores);

                for (var i = 0; i < isDone.Length; ++i)
                {
                    isDone[i] = isDone[i] || criterionDone[i];
                }
            }

            if (isDone.All(e => e))
            {
                results.AddRange(allInputIds.Select(allInputId => DecodeSingle(_tokenizer, decoder, allInputId.Select(Convert.ToInt32).ToArray())));
                break;
            }
            else
            {
                decoderInputIds = new DenseTensor<long>(generatedInputIds, dimensions: new int[] { generatedInputIds.Length, 1 });
            }
        }

        return results;
    }


    private (DenseTensor<long> inputIds, DenseTensor<long> attentionMask) GetTextInputs(string[] sentences)
    {
        var numSentences = sentences.Length;

        var encoded = _tokenizer.Encode(sentences);

        var tokenCount = encoded.First().InputIds.Length;

        var inputIds             = new long[encoded.Sum(s => s.InputIds.Length)];
        var flattenAttentionMask = new long[encoded.Sum(s => s.AttentionMask.Length)];

        var flattenInputIDsSpan      = inputIds.AsSpan();
        var flattenAttentionMaskSpan = flattenAttentionMask.AsSpan();

        foreach (var (InputIds, AttentionMask) in encoded)
        {
            InputIds.AsSpan().CopyTo(flattenInputIDsSpan);
            flattenInputIDsSpan = flattenInputIDsSpan.Slice(InputIds.Length);

            AttentionMask.AsSpan().CopyTo(flattenAttentionMaskSpan);
            flattenAttentionMaskSpan = flattenAttentionMaskSpan.Slice(AttentionMask.Length);
        }

        var dimensions = new[] { numSentences, tokenCount };

        return (inputIds: new DenseTensor<long>(inputIds, dimensions), attentionMask: new DenseTensor<long>(flattenAttentionMask, dimensions));

    }

    private static (DenseTensor<float> inputs_embeds, DenseTensor<long> attentionMask) MergeInputIdsWithImageFeatures(
        DenseTensor<float> inputsEmbeds,
        DenseTensor<float> imageFeatures,
        DenseTensor<long>  attentionMask
    )
    {
        return (
            inputs_embeds: TensorExtension.ConcatTensor(
                imageFeatures, // image embeds
                inputsEmbeds, // task prefix embeds
                axis: 1),
            attentionMask: TensorExtension.ConcatTensor(
                TensorExtension.OnesLong(imageFeatures.Dimensions.Slice(0, 2)), // image attention mask
                attentionMask, // task prefix attention mask
                axis: 1));
    }

    private static string DecodeSingle(Florence2Tokenizer tokenizer, ByteLevelDecoder decoder, int[] token_ids)
    {
        var tokens = token_ids.Select(tokenizer.IdToToken);

        var decoded = string.Join(string.Empty, decoder.DecodeChain(tokenizer, tokens.ToArray()));
        decoded = CleanUpTokenization(decoded);

        return decoded;
    }

    private static string CleanUpTokenization(string text)
    {
        // Clean up a list of simple English tokenization artifacts
        // like spaces before punctuations and abbreviated forms
        return text.Replace(" .", ".")
           .Replace(" ?",   "?")
           .Replace(" !",   "!")
           .Replace(" ,",   ",")
           .Replace(" ' ",  "")
           .Replace(" n't", "n't")
           .Replace(" 'm",  "'m")
           .Replace(" 's",  "'s")
           .Replace(" 've", "'ve")
           .Replace(" 're", "'re");
    }


    private IEnumerable<NamedOnnxValue> FromPresent(IDisposableReadOnlyCollection<DisposableNamedOnnxValue> decoderOut, bool useCache, NamedOnnxValue[]? pastKeyValues)
    {
        foreach (var decoderOutput in decoderOut)
        {
            if (decoderOutput.Name.StartsWith("present"))
            {
                var newName = decoderOutput.Name.Replace("present", "past_key_values");

                if (useCache && decoderOutput.Name.Contains("encoder"))
                {
                    //use cache
                    var vec = pastKeyValues.First(e => e.Name == newName).Value as DenseTensor<float>;
                    if (vec is null) throw new InvalidOperationException();
                    yield return NamedOnnxValue.CreateFromTensor(newName, vec.Clone());
                }
                else
                {
                    var vec = decoderOutput.Value as DenseTensor<float>;
                    if (vec is null) throw new InvalidOperationException();
                    yield return NamedOnnxValue.CreateFromTensor(newName, vec.Clone());
                }
            }
        }
    }

    private IEnumerable<NamedOnnxValue> InitPastKeyValues(NormalizedConfig normalizedConfig)
    {
        var prefix    = "past_key_values";
        var batchSize = 1;

        var encoderDimKv = normalizedConfig.EncoderHiddenSize / normalizedConfig.NumEncoderHeads;

        var decoderDimKv = normalizedConfig.DecoderHiddenSize / normalizedConfig.NumDecoderHeads;

        var encoderDims = new[] { batchSize, normalizedConfig.NumDecoderHeads, 0, encoderDimKv };
        var decoderDims = new[] { batchSize, normalizedConfig.NumDecoderHeads, 0, decoderDimKv };

        for (var i = 0; i < normalizedConfig.NumDecoderLayers; ++i)
        {
            yield return NamedOnnxValue.CreateFromTensor($"{prefix}.{i}.encoder.key",   new DenseTensor<float>(encoderDims));
            yield return NamedOnnxValue.CreateFromTensor($"{prefix}.{i}.encoder.value", new DenseTensor<float>(encoderDims));
            yield return NamedOnnxValue.CreateFromTensor($"{prefix}.{i}.decoder.key",   new DenseTensor<float>(decoderDims));
            yield return NamedOnnxValue.CreateFromTensor($"{prefix}.{i}.decoder.value", new DenseTensor<float>(decoderDims));
        }
    }

    public static Dictionary<TaskTypes, string> TaskPromptsWithoutInputsDict = new Dictionary<TaskTypes, string>
    {
        { TaskTypes.OCR, "What is the text in the image?" },
        { TaskTypes.OCR_WITH_REGION, "What is the text in the image, with regions?" },
        { TaskTypes.CAPTION, "What does the image describe?" },
        { TaskTypes.DETAILED_CAPTION, "Describe in detail what is shown in the image." },
        { TaskTypes.MORE_DETAILED_CAPTION, "Describe with a paragraph what is shown in the image." },
        { TaskTypes.OD, "Locate the objects with category name in the image." },
        { TaskTypes.DENSE_REGION_CAPTION, "Locate the objects in the image, with their descriptions." },
        { TaskTypes.REGION_PROPOSAL, "Locate the region proposals in the image." }
    };

    public Dictionary<TaskTypes, string> TaskPromptsWithInputDict = new Dictionary<TaskTypes, string>
    {
        { TaskTypes.CAPTION_TO_PHRASE_GROUNDING, "Locate the phrases in the caption: {0}" },
        { TaskTypes.REFERRING_EXPRESSION_SEGMENTATION, "Locate {0} in the image with mask" },
        { TaskTypes.REGION_TO_SEGMENTATION, "What is the polygon mask of region {0}" },
        { TaskTypes.OPEN_VOCABULARY_DETECTION, "Locate {0} in the image." },
        { TaskTypes.REGION_TO_CATEGORY, "What is the region {0}?" },
        { TaskTypes.REGION_TO_DESCRIPTION, "What does the region {0} describe?" },
        { TaskTypes.REGION_TO_OCR, "What text is in the region {0}?" },
    };
}