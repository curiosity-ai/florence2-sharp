using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Florence2;

public interface ILogitsSampler
{

    IEnumerable<(long token, double score)> Sample(int batchIdx, DenseTensor<float> logits);
}
class BeamSearchSampler : ILogitsSampler
{

    private readonly int              top_k;
    private readonly int              num_beams;
    private readonly InferenceSession topKSession;

    public BeamSearchSampler(InferenceSession topKSession, int topK, int numBeams)
    {
        this.topKSession = topKSession;
        top_k            = topK;
        num_beams        = numBeams;
    }

    public IEnumerable<(long token, double score)> Sample(int batchIdx, DenseTensor<float> logits)
    {
        var k = logits.Dimensions[logits.Dimensions.Length - 1]; // defaults to vocab size

        if (this.top_k > 0)
        {
            k = Math.Min(this.top_k, k);
        }

        var batchSize  = logits.Dimensions[0];
        var dimensions = logits.Dimensions.ToArray();
        dimensions[0] = 1;

        var newLength = dimensions[1];
        var start     = batchIdx * newLength;

        var logitsBatch = new DenseTensor<float>(logits.Buffer.Slice(start, newLength), dimensions);

        var result = TensorOperationRegistry.CallTopK(topKSession, logitsBatch, new DenseTensor<long>(new long[] { k }, new int[] { 1 }));
        var v      = result.First(v => v.Name == "v").AsTensor<float>().ToDenseTensor().ToArray();
        var i      = result.First(v => v.Name == "i").AsTensor<long>().ToDenseTensor().ToArray();

        // Compute softmax over logits
        var probabilities = Softmax(v.ToArray());

        for (int x = 0; x < num_beams; x++)
        {
            yield return (
                token: i[x], // token id
                score: Math.Log(probabilities[x]) // score
            );
        }

    }

    public static float[] Softmax(float[] arr)
    {
        // Compute the maximum value in the array
        var maxVal = arr.Max();

        // Compute the exponentials of the array values
        double[] exps = arr.Select(x => Math.Exp(x - maxVal)).ToArray();

        // Compute the sum of the exponentials
        double sumExps = exps.Sum();

        // Compute the softmax values
        double[] softmaxArr = exps.Select(x => x / sumExps).ToArray();

        return softmaxArr.Select(e => (float)e).ToArray();
    }
}