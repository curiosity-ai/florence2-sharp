using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;

namespace BERTTokenizers.Base
{
    public abstract class TokenizerBase
    {
        protected readonly List<string>            _vocabulary;
        protected readonly Dictionary<string, int> _vocabularyDict;
        public readonly    Tokens                  Tokens;

        public static int MaxWordLength = 50;

        public static void SetMaxWordLength(int maxWordLength)
        {
            MaxWordLength = maxWordLength;
        }

        public TokenizerBase(List<string> vocabulary, Tokens tokens)
        {
            _vocabulary = vocabulary;
            Tokens      = tokens;

            if (!_vocabulary.Any()) throw new Exception("vocab empty");

            _vocabularyDict = new Dictionary<string, int>();

            for (int i = 0; i < _vocabulary.Count; i++)
            {
                _vocabularyDict[_vocabulary[i]] = i;
            }
        }

        public List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> Encode(params string[] texts)
        {
            const int MaxTokens = 1024;
            var       tokenized = Tokenize(MaxTokens, texts);

            if (tokenized.Count == 0)
            {
                return new List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)>();
            }

            int sequenceLength = tokenized.Max(t => Math.Min(MaxTokens, t.Length));

            return tokenized.Select(tokens =>
            {
                var padding = Enumerable.Repeat(0L, sequenceLength - Math.Min(MaxTokens, tokens.Length)).ToList();

                var tokenIndexes   = tokens.Take(MaxTokens).Select(token => (long)token.VocabularyIndex).Concat(padding).ToArray();
                var segmentIndexes = tokens.Take(MaxTokens).Select(token => token.SegmentIndex).Concat(padding).ToArray();
                var inputMask      = tokens.Take(MaxTokens).Select(o => 1L).Concat(padding).ToArray();
                return (tokenIndexes, segmentIndexes, inputMask);
            }).ToList();
        }

        public string IdToToken(int id)
        {
            return _vocabulary[id];
        }

        public int TokenToID(string token)
        {
            return _vocabulary.IndexOf(token);
        }

        public List<string> Untokenize(List<string> tokens)
        {
            var currentToken = string.Empty;
            var untokens     = new List<string>();
            tokens.Reverse();

            tokens.ForEach(token =>
            {
                if (token.StartsWith("##"))
                {
                    currentToken = token.Replace("##", "") + currentToken;
                }
                else
                {
                    currentToken = token + currentToken;
                    untokens.Add(currentToken);
                    currentToken = string.Empty;
                }
            });

            untokens.Reverse();

            return untokens;
        }

        public List<(string Token, int VocabularyIndex, long SegmentIndex)[]> Tokenize(int maxTokens, params string[] texts)
        {
            return texts
               .Select(text =>
                {
                    var tokenAndIndex = new[] { Tokens.Classification }
                       .Concat(TokenizeSentence(text).Take(maxTokens))
                       .Concat(new[] { Tokens.Separation })
                       .SelectMany(TokenizeSubwords).Take(maxTokens);
                    var segmentIndexes = SegmentIndex(tokenAndIndex);

                    return tokenAndIndex.Zip(segmentIndexes, (tokenindex, segmentindex)
                        => (tokenindex.Token, tokenindex.VocabularyIndex, segmentindex)).ToArray();
                })
               .ToList();
        }

        private IEnumerable<long> SegmentIndex(IEnumerable<(string token, int index)> tokens)
        {
            var segmentIndex   = 0;
            var segmentIndexes = new List<long>();

            foreach (var (token, index) in tokens)
            {
                segmentIndexes.Add(segmentIndex);

                if (token == Tokens.Separation)
                {
                    segmentIndex++;
                }
            }

            return segmentIndexes;
        }

        private IEnumerable<(string Token, int VocabularyIndex)> TokenizeSubwords(string word)
        {
            if (word.Length > MaxWordLength) yield break; //Ignore words that are too long

            if (_vocabularyDict.TryGetValue(word, out var wordIndex))
            {
                yield return (word, wordIndex);
                yield break;
            }

            foreach (var inner in TokenizeSubwordsInner(word))
            {
                yield return inner;
            }
        }

        private List<(string token, int index)> TokenizeSubwordsInner(string word)
        {
            var tokens    = new List<(string token, int index)>();
            var remaining = word;

            while (!string.IsNullOrEmpty(remaining) && remaining.Length > 2)
            {
                string prefix        = null;
                int    subwordLength = remaining.Length;

                int stopLimit = remaining.StartsWith("##", StringComparison.Ordinal) ? 2 : 1;

                while (subwordLength >= stopLimit) // was initially 2, which prevents using "character encoding"
                {
                    string subword = remaining.Substring(0, subwordLength);

                    if (!_vocabularyDict.ContainsKey(subword))
                    {
                        subwordLength--;
                        continue;
                    }

                    prefix = subword;
                    break;
                }

                if (string.IsNullOrEmpty(prefix))
                {
                    tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown]));

                    return tokens;
                }

                //var regex = new Regex(prefix);
                //remaining = regex.Replace(remaining, "##", 1);


                var remainingAfter = ReplaceFirst(remaining, prefix, "##");

                if (remaining == remainingAfter)
                {
                    tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown]));

                    return tokens;
                }
                else
                {
                    remaining = remainingAfter;
                }

                tokens.Add((prefix, _vocabularyDict[prefix]));
            }

            if (!string.IsNullOrWhiteSpace(word) && !tokens.Any())
            {
                tokens.Add((Tokens.Unknown, _vocabularyDict[Tokens.Unknown]));
            }

            return tokens;
        }

        private static string ReplaceFirst(string text, string search, string replace)
        {
            int pos = text.IndexOf(search, StringComparison.Ordinal);

            if (pos < 0)
            {
                return text;
            }
            return text.Substring(0, pos) + replace + text.Substring(pos + search.Length);
        }

        protected abstract IEnumerable<string> TokenizeSentence(string text);
    }
}