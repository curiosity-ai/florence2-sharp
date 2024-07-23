namespace Florence2;

public static class GenerationConfig
{
    public static int NoRepeatNgramSize { get; set; } = 3;
    public static int NumBeams          { get; set; } = 3;
    public static int MaxLength         { get; set; } = 1025;
    public static int TopK              { get; set; } = 50;

}
public class NormalizedConfig
{
    public int NumDecoderLayers  { get; set; } = 6;
    public int NumDecoderHeads   { get; set; } = 12;
    public int DecoderHiddenSize { get; set; } = 768;
    public int NumEncoderLayers  { get; set; } = 6;
    public int NumEncoderHeads   { get; set; } = 12;
    public int EncoderHiddenSize { get; set; } = 768;
}