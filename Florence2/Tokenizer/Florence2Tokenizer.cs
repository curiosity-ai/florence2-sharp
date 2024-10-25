using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace Florence2;

public class Florence2Tokenizer
{

    protected readonly string[]                _vocabulary;
    protected readonly Dictionary<string, int> _vocabularyDict;
    public readonly    Tokens                  Tokens;

    public HashSet<string>                SpecialTokens { get; set; }
    public Dictionary<string, AddedToken> AddedTokens   { get; set; }

    private Florence2Tokenizer(Dictionary<string, int> vocabulary, Dictionary<string, AddedToken> addedTokens, Tokens tokens)
    {
        _vocabulary = new string[51289];

        foreach (var (token, index) in vocabulary)
        {
            if (_vocabulary[index] != null) throw new InvalidOperationException();

            _vocabulary[index] = token;
        }

        Tokens = tokens;

        foreach (var addedToken in addedTokens)
        {
            var id = int.Parse(addedToken.Key);

            if (_vocabulary[id] != null && _vocabulary[id] != addedToken.Value.Content) throw new InvalidOperationException();

            _vocabulary[id] = addedToken.Value.Content;
        }

        if (!_vocabulary.Any()) throw new Exception("vocab empty");

        _vocabularyDict = new Dictionary<string, int>();

        for (int i = 0; i < _vocabulary.Length; i++)
        {
            _vocabularyDict[_vocabulary[i]] = i;
        }

        SpecialTokens = addedTokens.Where(t => t.Value.Special).Select(kv => kv.Value.Content).ToHashSet();
        AddedTokens   = addedTokens;
        ByteToUnicode = UnicodeToBytes.ToDictionary(kv => kv.Value, kv => kv.Key);
    }

    public static Florence2Tokenizer Init()
    {
        var fileVocab = ResourceLoader.OpenResource(typeof(Florence2Tokenizer).Assembly, "vocab.json");
        var vocab     = JsonSerializer.Deserialize<Dictionary<string, int>>(fileVocab);

        var fileTokenizerConfig = ResourceLoader.OpenResource(typeof(Florence2Tokenizer).Assembly, "tokenizer_config.json");
        var tokenizerConfig     = JsonSerializer.Deserialize<TokenizerConfig>(fileTokenizerConfig);


        if (vocab is object && tokenizerConfig is object)
        {
            var maxID = vocab.Max(kv => kv.Value);

            var vocabList = new string[maxID + 1];

            foreach (var kv in vocab)
            {
                if (vocabList[kv.Value] != null) throw new NotImplementedException();
                vocabList[kv.Value] = kv.Key;
            }

            if (vocabList.Any(v => v == null)) throw new Exception("missing token");


            return new Florence2Tokenizer(vocab, tokenizerConfig.AddedTokensDecoder, new Tokens
            {
                Padding             = tokenizerConfig.PadToken,
                Unknown             = tokenizerConfig.UnkToken,
                Classification      = tokenizerConfig.ClsToken,
                Separation          = tokenizerConfig.SepToken,
                Mask                = tokenizerConfig.MaskToken,
                EndOfSequence       = tokenizerConfig.EosToken,
                BeginningOfSequence = tokenizerConfig.BosToken
            });
        }
        else
        {
            throw new Exception("vocab empty");
        }
    }

    private Regex _regex = new Regex("/'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+/", RegexOptions.ECMAScript);

    protected IEnumerable<string> TokenizeSentence(string text)
    {
        var tokens = _regex.Matches(text).Select(m => m.Value).ToArray();
        tokens = tokens.Select(t => new string(Encoding.UTF8.GetBytes(t).Select(b => ByteToUnicode[b]).ToArray())).ToArray();

        return tokens; //TODO make non regex
//        return text.Split(new string[] { " ", "   ", "\r\n" }, StringSplitOptions.None)
//           .SelectMany(o => SplitAndKeep(o, ".,;:\\/?!#$%()=+-*\"'–_`<>&^@{}[]|~'".ToArray()))
//           .Select(o => o.ToLower());
    }

    public static IEnumerable<string> SplitAndKeep(
        string        inputString,
        params char[] delimiters)
    {
        int start = 0, index;

        while ((index = inputString.IndexOfAny(delimiters, start)) != -1)
        {
            if (index - start > 0)
                yield return inputString.Substring(start, index - start);

            yield return inputString.Substring(index, 1);

            start = index + 1;
        }

        if (start < inputString.Length)
        {
            yield return inputString.Substring(start);
        }
    }


    public List<(long[] InputIds, long[] AttentionMask)> Encode(params string[] texts)
    {
        const int MaxTokens = 512; //Maximum token length supported by MiniLM model
        var       tokenized = Tokenize(texts);

        if (tokenized.Count == 0)
        {
            return new List<(long[] InputIds, long[] AttentionMask)>();
        }

        int sequenceLength = tokenized.Max(t => Math.Min(MaxTokens, t.Length));

        return tokenized.Select(tokens =>
        {
            var padding = Enumerable.Repeat(0L, sequenceLength - Math.Min(MaxTokens, tokens.Length)).ToList();

            var tokenIndexes = tokens.Take(MaxTokens).Select(token => (long)token.VocabularyIndex).Concat(padding).ToArray();
//            var segmentIndexes = tokens.Take(MaxTokens).Select(token => token.SegmentIndex).Concat(padding).ToArray();
            var inputMask = tokens.Take(MaxTokens).Select(o => 1L).Concat(padding).ToArray();
            return (tokenIndexes, inputMask);
        }).ToList();
    }

    public string IdToToken(int id)
    {
        return _vocabulary[id];
    }

    public List<string> ConvertIdsToTokens(IEnumerable<int> ids, bool skipSpecialTokens = false)
    {
        var result = ids.Select(IdToToken);

        if (skipSpecialTokens)
        {
            result = result.Where(token => !SpecialTokens.Contains(token));
        }
        return result.ToList();
    }


    public int TokenToID(string token)
    {
        return Array.IndexOf(_vocabulary, token);
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

    public List<(string Token, int VocabularyIndex)[]> Tokenize(params string[] texts)
    {
        var unkTokenId = _vocabularyDict[Tokens.Unknown];

        return texts
           .Select(text =>
                new[] { Tokens.Classification }
                   .Concat(TokenizeSentence(text))
                   .Concat(new[] { Tokens.Separation })
                   .Select(t => (t, _vocabularyDict.GetValueOrDefault(t, unkTokenId)))
                   .ToArray())
           .ToList();
    }

//    private IEnumerable<long> SegmentIndex(IEnumerable<(string token, int index)> tokens)
//    {
//        var segmentIndex   = 0;
//        var segmentIndexes = new List<long>();
//
//        foreach (var (token, index) in tokens)
//        {
//            segmentIndexes.Add(segmentIndex);
//
//            if (token == Tokens.Separation)
//            {
//                segmentIndex++;
//            }
//        }
//
//        return segmentIndexes;
//    }

    private static string ReplaceFirst(string text, string search, string replace)
    {
        int pos = text.IndexOf(search, StringComparison.Ordinal);

        if (pos < 0)
        {
            return text;
        }
        return text.Substring(0, pos) + replace + text.Substring(pos + search.Length);
    }

    public class TokenizerConfig
    {
        [JsonPropertyName("add_prefix_space")]             public bool                           AddPrefixSpace            { get; set; }
        [JsonPropertyName("added_tokens_decoder")]         public Dictionary<string, AddedToken> AddedTokensDecoder        { get; set; }
        [JsonPropertyName("additional_special_tokens")]    public string[]                       AdditionalSpecialTokens   { get; set; }
        [JsonPropertyName("bos_token")]                    public string                         BosToken                  { get; set; }
        [JsonPropertyName("clean_up_tokenization_spaces")] public bool                           CleanUpTokenizationSpaces { get; set; }
        [JsonPropertyName("cls_token")]                    public string                         ClsToken                  { get; set; }
        [JsonPropertyName("eos_token")]                    public string                         EosToken                  { get; set; }
        [JsonPropertyName("errors")]                       public string                         Errors                    { get; set; }
        [JsonPropertyName("mask_token")]                   public string                         MaskToken                 { get; set; }
        [JsonPropertyName("model_max_length")]             public long                           ModelMaxLength            { get; set; }
        [JsonPropertyName("pad_token")]                    public string                         PadToken                  { get; set; }
        [JsonPropertyName("processor_class")]              public string                         ProcessorClass            { get; set; }
        [JsonPropertyName("sep_token")]                    public string                         SepToken                  { get; set; }
        [JsonPropertyName("tokenizer_class")]              public string                         TokenizerClass            { get; set; }
        [JsonPropertyName("trim_offsets")]                 public bool                           TrimOffsets               { get; set; }
        [JsonPropertyName("unk_token")]                    public string                         UnkToken                  { get; set; }
    }

    public class AddedToken
    {
        [JsonPropertyName("content")]     public string Content    { get; set; }
        [JsonPropertyName("lstrip")]      public bool   Lstrip     { get; set; }
        [JsonPropertyName("normalized")]  public bool   Normalized { get; set; }
        [JsonPropertyName("rstrip")]      public bool   Rstrip     { get; set; }
        [JsonPropertyName("single_word")] public bool   SingleWord { get; set; }
        [JsonPropertyName("special")]     public bool   Special    { get; set; }
    }


    public readonly Dictionary<char, byte> UnicodeToBytes = new Dictionary<char, byte>()
    {
        { '0', 48 },
        { '1', 49 },
        { '2', 50 },
        { '3', 51 },
        { '4', 52 },
        { '5', 53 },
        { '6', 54 },
        { '7', 55 },
        { '8', 56 },
        { '9', 57 },
        { 'Ā', 0 },
        { 'ā', 1 },
        { 'Ă', 2 },
        { 'ă', 3 },
        { 'Ą', 4 },
        { 'ą', 5 },
        { 'Ć', 6 },
        { 'ć', 7 },
        { 'Ĉ', 8 },
        { 'ĉ', 9 },
        { 'Ċ', 10 },
        { 'ċ', 11 },
        { 'Č', 12 },
        { 'č', 13 },
        { 'Ď', 14 },
        { 'ď', 15 },
        { 'Đ', 16 },
        { 'đ', 17 },
        { 'Ē', 18 },
        { 'ē', 19 },
        { 'Ĕ', 20 },
        { 'ĕ', 21 },
        { 'Ė', 22 },
        { 'ė', 23 },
        { 'Ę', 24 },
        { 'ę', 25 },
        { 'Ě', 26 },
        { 'ě', 27 },
        { 'Ĝ', 28 },
        { 'ĝ', 29 },
        { 'Ğ', 30 },
        { 'ğ', 31 },
        { 'Ġ', 32 },
        { '!', 33 },
        { '\"', 34 },
        { '#', 35 },
        { '$', 36 },
        { '%', 37 },
        { '&', 38 },
        { '\'', 39 },
        { '(', 40 },
        { ')', 41 },
        { '*', 42 },
        { '+', 43 },
        { ',', 44 },
        { '-', 45 },
        { '.', 46 },
        { '/', 47 },
        { ':', 58 },
        { ';', 59 },
        { '<', 60 },
        { '=', 61 },
        { '>', 62 },
        { '?', 63 },
        { '@', 64 },
        { 'A', 65 },
        { 'B', 66 },
        { 'C', 67 },
        { 'D', 68 },
        { 'E', 69 },
        { 'F', 70 },
        { 'G', 71 },
        { 'H', 72 },
        { 'I', 73 },
        { 'J', 74 },
        { 'K', 75 },
        { 'L', 76 },
        { 'M', 77 },
        { 'N', 78 },
        { 'O', 79 },
        { 'P', 80 },
        { 'Q', 81 },
        { 'R', 82 },
        { 'S', 83 },
        { 'T', 84 },
        { 'U', 85 },
        { 'V', 86 },
        { 'W', 87 },
        { 'X', 88 },
        { 'Y', 89 },
        { 'Z', 90 },
        { '[', 91 },
        { '\\', 92 },
        { ']', 93 },
        { '^', 94 },
        { '_', 95 },
        { '`', 96 },
        { 'a', 97 },
        { 'b', 98 },
        { 'c', 99 },
        { 'd', 100 },
        { 'e', 101 },
        { 'f', 102 },
        { 'g', 103 },
        { 'h', 104 },
        { 'i', 105 },
        { 'j', 106 },
        { 'k', 107 },
        { 'l', 108 },
        { 'm', 109 },
        { 'n', 110 },
        { 'o', 111 },
        { 'p', 112 },
        { 'q', 113 },
        { 'r', 114 },
        { 's', 115 },
        { 't', 116 },
        { 'u', 117 },
        { 'v', 118 },
        { 'w', 119 },
        { 'x', 120 },
        { 'y', 121 },
        { 'z', 122 },
        { '{', 123 },
        { '|', 124 },
        { '}', 125 },
        { '~', 126 },
        { 'ġ', 127 },
        { 'Ģ', 128 },
        { 'ģ', 129 },
        { 'Ĥ', 130 },
        { 'ĥ', 131 },
        { 'Ħ', 132 },
        { 'ħ', 133 },
        { 'Ĩ', 134 },
        { 'ĩ', 135 },
        { 'Ī', 136 },
        { 'ī', 137 },
        { 'Ĭ', 138 },
        { 'ĭ', 139 },
        { 'Į', 140 },
        { 'į', 141 },
        { 'İ', 142 },
        { 'ı', 143 },
        { 'Ĳ', 144 },
        { 'ĳ', 145 },
        { 'Ĵ', 146 },
        { 'ĵ', 147 },
        { 'Ķ', 148 },
        { 'ķ', 149 },
        { 'ĸ', 150 },
        { 'Ĺ', 151 },
        { 'ĺ', 152 },
        { 'Ļ', 153 },
        { 'ļ', 154 },
        { 'Ľ', 155 },
        { 'ľ', 156 },
        { 'Ŀ', 157 },
        { 'ŀ', 158 },
        { 'Ł', 159 },
        { 'ł', 160 },
        { '¡', 161 },
        { '¢', 162 },
        { '£', 163 },
        { '¤', 164 },
        { '¥', 165 },
        { '¦', 166 },
        { '§', 167 },
        { '¨', 168 },
        { '©', 169 },
        { 'ª', 170 },
        { '«', 171 },
        { '¬', 172 },
        { 'Ń', 173 },
        { '®', 174 },
        { '¯', 175 },
        { '°', 176 },
        { '±', 177 },
        { '²', 178 },
        { '³', 179 },
        { '´', 180 },
        { 'µ', 181 },
        { '¶', 182 },
        { '·', 183 },
        { '¸', 184 },
        { '¹', 185 },
        { 'º', 186 },
        { '»', 187 },
        { '¼', 188 },
        { '½', 189 },
        { '¾', 190 },
        { '¿', 191 },
        { 'À', 192 },
        { 'Á', 193 },
        { 'Â', 194 },
        { 'Ã', 195 },
        { 'Ä', 196 },
        { 'Å', 197 },
        { 'Æ', 198 },
        { 'Ç', 199 },
        { 'È', 200 },
        { 'É', 201 },
        { 'Ê', 202 },
        { 'Ë', 203 },
        { 'Ì', 204 },
        { 'Í', 205 },
        { 'Î', 206 },
        { 'Ï', 207 },
        { 'Ð', 208 },
        { 'Ñ', 209 },
        { 'Ò', 210 },
        { 'Ó', 211 },
        { 'Ô', 212 },
        { 'Õ', 213 },
        { 'Ö', 214 },
        { '×', 215 },
        { 'Ø', 216 },
        { 'Ù', 217 },
        { 'Ú', 218 },
        { 'Û', 219 },
        { 'Ü', 220 },
        { 'Ý', 221 },
        { 'Þ', 222 },
        { 'ß', 223 },
        { 'à', 224 },
        { 'á', 225 },
        { 'â', 226 },
        { 'ã', 227 },
        { 'ä', 228 },
        { 'å', 229 },
        { 'æ', 230 },
        { 'ç', 231 },
        { 'è', 232 },
        { 'é', 233 },
        { 'ê', 234 },
        { 'ë', 235 },
        { 'ì', 236 },
        { 'í', 237 },
        { 'î', 238 },
        { 'ï', 239 },
        { 'ð', 240 },
        { 'ñ', 241 },
        { 'ò', 242 },
        { 'ó', 243 },
        { 'ô', 244 },
        { 'õ', 245 },
        { 'ö', 246 },
        { '÷', 247 },
        { 'ø', 248 },
        { 'ù', 249 },
        { 'ú', 250 },
        { 'û', 251 },
        { 'ü', 252 },
        { 'ý', 253 },
        { 'þ', 254 },
        { 'ÿ', 255 }
    };

    public readonly Dictionary<byte, char> ByteToUnicode;

}