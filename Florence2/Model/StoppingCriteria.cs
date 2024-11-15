using System.Collections.Generic;
using System.Linq;

namespace Florence2;

public interface StoppingCriteria
{
    public bool[] Call(List<long>[] inputIds, double[] scores);
}
public class MaxLengthCriteria : StoppingCriteria
{
    private readonly int maxLength;

    public MaxLengthCriteria(int max_length)
    {
        this.maxLength = max_length;
    }

    public bool[] Call(List<long>[] inputIds, double[] scores)
    {
        return inputIds.Select(ids => ids.Count >= this.maxLength).ToArray();
    }
}
public class EosTokenCriteria : StoppingCriteria
{
    private readonly long eosTokenID;

    public EosTokenCriteria(long eosTokenID)
    {
        this.eosTokenID = eosTokenID;
    }

    public bool[] Call(List<long>[] inputIds, double[] scores)
    {
        return inputIds.Select(ids => ids.Last() == eosTokenID).ToArray();
    }
}