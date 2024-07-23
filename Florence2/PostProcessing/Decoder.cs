using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Florence2;

public class Decoder
{

}
public class ByteLevelDecoder : Decoder
{

    private Dictionary<string, Florence2Tokenizer.AddedToken> added_tokens;

    public ByteLevelDecoder(Dictionary<string, Florence2Tokenizer.AddedToken> added_tokens)
    {
        this.added_tokens = added_tokens;
    }

    private string convert_tokens_to_string(Florence2Tokenizer tokenizer, IEnumerable<string> tokens)
    {
        var    text         = string.Join("", tokens);
        byte[] byteArray    = text.Select(c => tokenizer.UnicodeToBytes[c]).ToArray();
        var    decoded_text = Encoding.UTF8.GetString(byteArray);
        return decoded_text;
    }

    public List<string> DecodeChain(Florence2Tokenizer tokenizer, IEnumerable<string> tokens)
    {
        var sub_texts        = new List<string>();
        var current_sub_text = new List<string>();

        foreach (var token in tokens)
        {
            if (this.added_tokens.Any(x => x.Value.Content == token))
            {
                if (current_sub_text.Count > 0)
                {
                    sub_texts.Add(this.convert_tokens_to_string(tokenizer, current_sub_text));
                    current_sub_text = [];
                }
                sub_texts.Add(token);
            }
            else
            {
                current_sub_text.Add(token);
            }
        }

        if (current_sub_text.Count > 0)
        {
            sub_texts.Add(this.convert_tokens_to_string(tokenizer, current_sub_text));
        }

        // TODO add spaces_between_special_tokens and clean_up_tokenization_spaces options

        return sub_texts;
    }


}