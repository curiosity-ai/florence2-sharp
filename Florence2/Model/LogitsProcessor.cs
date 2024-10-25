using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Florence2;

public interface LogitsProcessor
{
    public void Process(int batchID, long[] input_ids, DenseTensor<float> logits);

    public static Span<T> GetBatchSlice<T>(int batchID, DenseTensor<T> tensor)
    {
        if (tensor.Dimensions.Length == 2)
        {
            return tensor.Buffer.Span.Slice(batchID * tensor.Dimensions[1], tensor.Dimensions[1]);
        }
        else if (tensor.Dimensions.Length == 3 && tensor.Dimensions[1] == 1)
        {
            return tensor.Buffer.Span.Slice(batchID * tensor.Dimensions[2], tensor.Dimensions[2]);
        }
        else
        {
            throw new NotImplementedException();
        }

    }
}
/**
 * A logits processor that disallows ngrams of a certain size to be repeated.
 */
public class NoRepeatNGramLogitsProcessor : LogitsProcessor
{
    private int noRepeatNgramSize;

    /**
     * Create a NoRepeatNGramLogitsProcessor.
     * @param no_repeat_ngram_size The no-repeat-ngram size. All ngrams of this size can only occur once.
     */
    public NoRepeatNGramLogitsProcessor(int no_repeat_ngram_size)
    {
        this.noRepeatNgramSize = no_repeat_ngram_size;
    }

    /**
     * Generate n-grams from a sequence of token ids.
     * @param prevInputIds List of previous input ids
     * @returns Map of generated n-grams
     */
    private Dictionary<long[], long[]> GetNgrams(int batchID, long[] prevInputIds)
    {
        long curLen = prevInputIds.Length;

        List<long[]> ngrams = new List<long[]>();

        for (int j = 0; j < curLen + 1 - this.noRepeatNgramSize; ++j)
        {
            long[] ngram = new long[this.noRepeatNgramSize];

            for (int k = 0; k < this.noRepeatNgramSize; ++k)
            {
                ngram[k] = prevInputIds[j + k];
            }
            ngrams.Add(ngram);
        }

        Dictionary<long[], long[]> generatedNgram = new Dictionary<long[], long[]>(new NGramEqualityComparer());

        foreach (long[] ngram in ngrams)
        {
            long[] prevNgram = ngram[0..(ngram.Length - 1)];

            long[] prevNgramKey   = prevNgram.ToArray();
            long[] prevNgramValue = generatedNgram.GetValueOrDefault(prevNgramKey, new long[] { });
            prevNgramValue               = prevNgramValue.Concat(new[] { ngram[ngram.Length - 1] }).ToArray();
            generatedNgram[prevNgramKey] = prevNgramValue;
        }
        return generatedNgram;
    }

    private class NGramEqualityComparer : IEqualityComparer<long[]>
    {
        public bool Equals(long[]? x, long[]? y)
        {
            if (x is null        && y is null) return true;
            else if (x is object && y is object) return x.SequenceEqual(y);
            else return false;
        }

        public int GetHashCode(long[] obj)
        {
            HashCode hash = new();

            foreach (var v in obj)
            {
                hash.Add(v);
            }
            return hash.ToHashCode();
        }
    }

    /**
     * Generate n-grams from a sequence of token ids.
     * @param bannedNgrams Map of banned n-grams
     * @param prevInputIds List of previous input ids
     * @returns Map of generated n-grams
     */
    private long[] GetGeneratedNgrams(int batchID, Dictionary<long[], long[]> bannedNgrams, long[] prevInputIds)
    {
        long[] ngramIdx = prevInputIds[(prevInputIds.Length + 1 - noRepeatNgramSize)..prevInputIds.Length].ToArray();
        var    banned   = bannedNgrams.GetValueOrDefault(ngramIdx, new long[] { });
        return banned;

//        long[] ngramKey = prevInputIds.Skip(((int)prevInputIds.Length) + 1 - this.noRepeatNgramSize).Take(this.noRepeatNgramSize).ToArray();
//        return bannedNgrams.GetValueOrDefault(ngramKey, new long[0]);
    }

    /**
     * Calculate banned n-gram tokens
     * @param prevInputIds List of previous input ids
     * @returns Map of generated n-grams
     */
    private long[] CalcBannedNgramTokens(int batchID, long[] prevInputIds)
    {
        if (prevInputIds.Length + 1 < this.noRepeatNgramSize)
        {
            // return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            return new long[] { };
        }
        else
        {
            Dictionary<long[], long[]> generatedNgrams = this.GetNgrams(batchID, prevInputIds);
            long[]                     bannedTokens    = this.GetGeneratedNgrams(batchID, generatedNgrams, prevInputIds);
            return bannedTokens;
        }
    }

    public void Process(int batchID, long[] input_ids, DenseTensor<float> logits)
    {
        long[] bannedTokens = this.CalcBannedNgramTokens(batchID, input_ids);

        foreach (int token in bannedTokens)
        {
            logits[batchID, token] = float.NegativeInfinity;
        }
    }
}

public class ForcedBOSTokenLogitsProcessor : LogitsProcessor
{
    private int bosTokenID;

    public ForcedBOSTokenLogitsProcessor(int bos_token_id)
    {
        this.bosTokenID = bos_token_id;
    }

    public void Process(int batchID, long[] input_ids, DenseTensor<float> logits)
    {
        if (input_ids.Length == 1)
        {
            LogitsProcessor.GetBatchSlice(batchID, logits).Fill(float.NegativeInfinity);
            logits[batchID, this.bosTokenID] = 0;
        }
    }
}
public class ForcedEOSTokenLogitsProcessor : LogitsProcessor
{

    private int max_length;
    private int eos_token_id;


    public ForcedEOSTokenLogitsProcessor(int max_length, int eos_token_id)
    {
        this.max_length   = max_length;
        this.eos_token_id = eos_token_id;
    }

    public void Process(int batchID, long[] input_ids, DenseTensor<float> logits)
    {
        if (input_ids.Length == this.max_length - 1)
        {
            LogitsProcessor.GetBatchSlice(batchID, logits).Fill(float.NegativeInfinity);
            logits[batchID, this.eos_token_id] = 0;
        }
    }
}