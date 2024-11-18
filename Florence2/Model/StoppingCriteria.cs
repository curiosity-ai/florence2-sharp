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
    private readonly long[] eosTokenID;

    public EosTokenCriteria(long[] eosTokenID)
    {
        this.eosTokenID = eosTokenID;
    }

    public EosTokenCriteria(long eosTokenID)
    {
        this.eosTokenID = new long[] { eosTokenID };
    }

    public bool[] Call(List<long>[] inputIds, double[] scores)
    {
        return inputIds.Select(ids =>
        {
            var last = ids.Last();

            return this.eosTokenID.Any(eosID => last == eosID);
        }).ToArray();
    }
}