namespace BERTTokenizers.Base
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
}