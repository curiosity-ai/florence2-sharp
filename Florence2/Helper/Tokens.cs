namespace Florence2
{
    public class Tokens
    {
        public string Padding             { get; init; }
        public string Unknown             { get; init; }
        public string Classification      { get; init; }
        public string Separation          { get; init; }
        public string Mask                { get; init; }
        public string EndOfSequence       { get; init; }
        public string BeginningOfSequence { get; init; }
    }

    public class SentanceTransformerTokens : Tokens
    {
        public new string Padding             => "";
        public new string Unknown             => "[UNK]";
        public new string Classification      => "[CLS]";
        public new string Separation          => "[SEP]";
        public new string Mask                => "[MASK]";
        public new string EndOfSequence       => null;
        public new string BeginningOfSequence => null;
    }

}