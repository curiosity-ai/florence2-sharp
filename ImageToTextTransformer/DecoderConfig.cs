namespace ImageToTextTransformer;

public static class GenerationConfig
{
    public static bool   FromModelConfig     { get; set; } = true;
    public static long   BosTokenId          { get; set; } = 0;
    public static long   DecoderStartTokenId { get; set; } = 2;
    public static bool   EarlyStopping       { get; set; } = true;
    public static long   EosTokenId          { get; set; } = 2;
    public static long   ForcedBosTokenId    { get; set; } = 0;
    public static long   ForcedEosTokenId    { get; set; } = 2;
    public static int    NoRepeatNgramSize   { get; set; } = 3;
    public static int    NumBeams            { get; set; } = 3;
    public static int    PadTokenId          { get; set; } = 1;
    public static int    MaxLength           { get; set; } = 1025;
    public static string TransformersVersion { get; set; } = "4.38.2";
    public static int    TopK                { get; set; } = 50;

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