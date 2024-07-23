using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Florence2;

public static class TensorOperationRegistry
{
    public static InferenceSession TopKSession(SessionOptions sessionOptions)
    {
        //see tensorflow.js 
        var sessionBytes = new byte[] { 8, 10, 18, 0, 58, 73, 10, 18, 10, 1, 120, 10, 1, 107, 18, 1, 118, 18, 1, 105, 34, 4, 84, 111, 112, 75, 18, 1, 116, 90, 9, 10, 1, 120, 18, 4, 10, 2, 8, 1, 90, 15, 10, 1, 107, 18, 10, 10, 8, 8, 7, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 118, 18, 4, 10, 2, 8, 1, 98, 9, 10, 1, 105, 18, 4, 10, 2, 8, 7, 66, 2, 16, 21 };
        var session      = new InferenceSession(sessionBytes, sessionOptions);
        return session;
    }

    public static IDisposableReadOnlyCollection<DisposableNamedOnnxValue> CallTopK(InferenceSession session, DenseTensor<float> x, DenseTensor<long> k)
    {
        var outputs = session.Run(new[] { NamedOnnxValue.CreateFromTensor("k", k), NamedOnnxValue.CreateFromTensor("x", x) });
        return outputs;
    }


}