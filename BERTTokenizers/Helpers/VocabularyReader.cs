using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace BERTTokenizers.Helpers
{
    public class VocabularyReader
    {
        public static List<string> ReadFileText(Stream vocabularyFile)
        {
            var result = new List<string>();

            using (var reader = new StreamReader(vocabularyFile))
            {
                string line;

                while ((line = reader.ReadLine()) != null)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        result.Add(line);
                    }
                }
            }

            return result;
        }

        public static async Task<List<string>> ReadFileJson(Stream vocabularyFile)
        {
            var vocab = await JsonSerializer.DeserializeAsync<Dictionary<string, int>>(vocabularyFile);
            return vocab?.OrderBy(kv => kv.Value).Select(kv => kv.Key).ToList() ?? new List<string>();
        }
    }
}