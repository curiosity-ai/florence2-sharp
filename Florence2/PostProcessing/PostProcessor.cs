using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace Florence2;

public class Florence2PostProcessor
{
    private Florence2PostProcessorConfig               config;
    private Dictionary<PostProcessingTypes, ParseTask> parseTaskConfigs;
    private BoxQuantizer                               boxQuantizer;
    private CoordinatesQuantizer                       coordinatesQuantizer;
    private HashSet<string>                            blackListOfPhraseGrounding;

    public Florence2PostProcessor()
    {
        parseTaskConfigs = new Dictionary<PostProcessingTypes, ParseTask>();
        config           = new Florence2PostProcessorConfig();

        foreach (var task in config.PARSE_TASKS)
        {
            parseTaskConfigs[task.TASK_NAME] = task;
        }

        boxQuantizer         = new BoxQuantizer(config.BOX_QUANTIZATION_MODE, (config.NUM_BBOX_WIDTH_BINS, config.NUM_BBOX_HEIGHT_BINS));
        coordinatesQuantizer = new CoordinatesQuantizer(config.BOX_QUANTIZATION_MODE, (config.COORDINATES_WIDTH_BINS, config.COORDINATES_HEIGHT_BINS));

        blackListOfPhraseGrounding = new HashSet<string>
        {
            "it", "I", "me", "mine",
            "you", "your", "yours",
            "he", "him", "his",
            "she", "her", "hers",
            "they", "them", "their", "theirs",
            "one", "oneself",
            "we", "us", "our", "ours",
            "you", "your", "yours",
            "they", "them", "their", "theirs",
            "mine", "yours", "his", "hers", "its",
            "ours", "yours", "theirs",
            "myself", "yourself", "himself", "herself", "itself",
            "ourselves", "yourselves", "themselves",
            "this", "that",
            "these", "those",
            "who", "whom", "whose", "which", "what",
            "who", "whom", "whose", "which", "that",
            "all", "another", "any", "anybody", "anyone", "anything",
            "each", "everybody", "everyone", "everything",
            "few", "many", "nobody", "none", "one", "several",
            "some", "somebody", "someone", "something",
            "each other", "one another",
            "myself", "yourself", "himself", "herself", "itself",
            "ourselves", "yourselves", "themselves",
            "the image", "image", "images", "the", "a", "an", "a group",
            "other objects", "lots", "a set"
        };
    }

    public enum PostProcessingTypes
    {
        od,
        ocr_with_region,
        pure_text,
        description_with_polygons,
        description_with_bboxes,
        phrase_grounding,
        polygons,
        description_with_bboxes_or_polygons,
        bboxes
    }


    public static PostProcessingTypes GetPostProcessingType(TaskTypes taskTypes)
    {
        return taskTypes switch
        {
            TaskTypes.OCR                               => PostProcessingTypes.pure_text,
            TaskTypes.OCR_WITH_REGION                   => PostProcessingTypes.ocr_with_region,
            TaskTypes.CAPTION                           => PostProcessingTypes.pure_text,
            TaskTypes.DETAILED_CAPTION                  => PostProcessingTypes.pure_text,
            TaskTypes.MORE_DETAILED_CAPTION             => PostProcessingTypes.pure_text,
            TaskTypes.OD                                => PostProcessingTypes.description_with_bboxes,
            TaskTypes.DENSE_REGION_CAPTION              => PostProcessingTypes.description_with_bboxes,
            TaskTypes.CAPTION_TO_PHRASE_GROUNDING       => PostProcessingTypes.phrase_grounding,
            TaskTypes.REFERRING_EXPRESSION_SEGMENTATION => PostProcessingTypes.polygons,
            TaskTypes.REGION_TO_SEGMENTATION            => PostProcessingTypes.polygons,
            TaskTypes.OPEN_VOCABULARY_DETECTION         => PostProcessingTypes.description_with_bboxes_or_polygons,
            TaskTypes.REGION_TO_CATEGORY                => PostProcessingTypes.pure_text,
            TaskTypes.REGION_TO_DESCRIPTION             => PostProcessingTypes.pure_text,
            TaskTypes.REGION_TO_OCR                     => PostProcessingTypes.pure_text,
            TaskTypes.REGION_PROPOSAL                   => PostProcessingTypes.bboxes,
            _                                           => PostProcessingTypes.pure_text
        };
    }

    public class Florence2PostProcessorConfig
    {
        public int           NUM_BBOX_HEIGHT_BINS          { get; set; } = 1000;
        public int           NUM_BBOX_WIDTH_BINS           { get; set; } = 1000;
        public QuantizerMode BOX_QUANTIZATION_MODE         { get; set; } = QuantizerMode.Floor;
        public int           COORDINATES_HEIGHT_BINS       { get; set; } = 1000;
        public int           COORDINATES_WIDTH_BINS        { get; set; } = 1000;
        public QuantizerMode COORDINATES_QUANTIZATION_MODE { get; set; } = QuantizerMode.Floor;
        public List<ParseTask> PARSE_TASKS { get;                 set; } = new List<ParseTask>
        {
            new ParseTask
            {
                TASK_NAME = PostProcessingTypes.od,
            },
            new ParseTask
            {
                TASK_NAME = PostProcessingTypes.ocr_with_region,
            },
            new ParseTask
            {
                TASK_NAME            = PostProcessingTypes.phrase_grounding,
                FILTER_BY_BLACK_LIST = true
            },
            new ParseTask { TASK_NAME = PostProcessingTypes.pure_text },
            new ParseTask { TASK_NAME = PostProcessingTypes.description_with_bboxes },
            new ParseTask { TASK_NAME = PostProcessingTypes.description_with_polygons },
            new ParseTask { TASK_NAME = PostProcessingTypes.polygons },
            new ParseTask { TASK_NAME = PostProcessingTypes.bboxes },
            new ParseTask { TASK_NAME = PostProcessingTypes.description_with_bboxes_or_polygons }
        };
    }

    public class ParseTask
    {
        public PostProcessingTypes TASK_NAME            { get; set; }
        public bool                FILTER_BY_BLACK_LIST { get; set; } = false;
    }

    public List<LabeledOCRBox> ParseOcrFromTextAndSpans(
        string                  text,
        (int width, int height) imageSize)
    {
        var pattern = @"(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>";

        var instances = new List<LabeledOCRBox>();
        text = text.Replace("<s>", "");

        // OCR with regions
        var parsed = Regex.Matches(text, pattern);

        foreach (Match ocrLine in parsed)
        {
            string ocrContent = ocrLine.Groups[1].Value;
            var    quadBox    = new List<int>();

            for (int i = 2; i <= 9; i++)
            {
                quadBox.Add(int.Parse(ocrLine.Groups[i].Value));
            }

            var dequantizedQuadBox = coordinatesQuantizer.Dequantize(quadBox.Chunk(2).Select(c => new Coordinates<int>(c)).ToArray(), imageSize).ToArray();

            instances.Add(new LabeledOCRBox
            {
                QuadBox = dequantizedQuadBox,
                Text    = ReplaceStartAndEndToken(ocrContent)
            });
        }

        return instances;
    }


    public IEnumerable<LabeledBoundingBoxes> ParsePhraseGroundingFromTextAndSpans(string text, (int width, int height) imageSize)
    {
        // Ignore <s> </s> and <pad>
        text = text.Replace("<s>", "").Replace("</s>", "").Replace("<pad>", "");

        var phrasePattern = @"([^<]+(?:<loc_\d+>){4,})";
        var phrases       = Regex.Matches(text, phrasePattern);

        var textPattern = @"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)";
        var boxPattern  = @"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>";


        foreach (Match phraseMatch in phrases)
        {
            var box = new LabeledBoundingBoxes();

            var phraseText = phraseMatch.Value;

            var phraseTextStrip = phraseText.Replace("<ground>", "", StringComparison.OrdinalIgnoreCase).Replace("<obj>", "", StringComparison.OrdinalIgnoreCase);

            if (string.IsNullOrWhiteSpace(phraseTextStrip))
            {
                continue;
            }

            var phraseMatch1 = Regex.Match(phraseTextStrip, textPattern);

            if (!phraseMatch1.Success)
            {
                continue;
            }

            var bboxesParsed = Regex.Matches(phraseText, boxPattern);

            if (bboxesParsed.Count == 0)
            {
                continue;
            }

            var phrase = phraseMatch1.Groups[1].Value.Trim();

            if (blackListOfPhraseGrounding.Contains(phrase))
            {
                continue;
            }

            var bboxBins = bboxesParsed.Select(match => new BoundingBox<int>(Enumerable.Range(1, 4).Select(i => int.Parse(match.Groups[i].Value)).ToArray())).ToArray();

            box.BBoxes = boxQuantizer.Dequantize(bboxBins, imageSize);

            // Exclude non-ASCII characters
            phrase = new string(phrase.Where(c => c < 128).ToArray());

            box.Label = ReplaceStartAndEndToken(phrase);
            yield return box;
        }
    }

    public IEnumerable<LabeledBoundingBoxes> ParseDescriptionWithBboxesFromTextAndSpans(string text, (int, int) imageSize, bool allowEmptyPhrase = false)
    {
        var patternQustionMark = @"([a-zA-Z0-9 ]+)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>";

        // Ignore <s> </s> and <pad>
        text = text.Replace("<s>", "").Replace("</s>", "").Replace("<pad>", "");

        var phrasePattern = allowEmptyPhrase ? @"(?:(?:<loc_\d+>){4,})" : @"([^<]+(?:<loc_\d+>){4,})";

        var phrases = Regex.Matches(text, phrasePattern);

        // Pattern should be text pattern and od pattern
        var textPattern = @"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_)";
        var boxPattern  = @"<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>";

        foreach (Match phraseText in phrases)
        {
            var box = new LabeledBoundingBoxes();

            var phraseTextStrip = phraseText.Value
               .Replace("<ground>", "", StringComparison.Ordinal)
               .Replace("<obj>",    "", StringComparison.Ordinal);

            if (string.IsNullOrEmpty(phraseTextStrip) && !allowEmptyPhrase)
            {
                continue;
            }

            // Parse phrase, get string
            var phraseMatch = Regex.Match(phraseTextStrip, textPattern);

            if (!phraseMatch.Success)
            {
                continue;
            }

            var phrase = phraseMatch.Groups[1].Value.Trim();

            // Parse bboxes by box_pattern
            var bboxesParsed = Regex.Matches(phraseText.Value, boxPattern);

            if (bboxesParsed.Count == 0)
            {
                continue;
            }

            // A list of lists
            BoundingBox<int>[] bboxBins = bboxesParsed.Select(m => new BoundingBox<int>(Enumerable.Range(1, 4).Select(i => int.Parse(m.Groups[i].Value)).ToArray())).ToArray();

            box.BBoxes = boxQuantizer.Dequantize(bboxBins, imageSize);

            // Exclude non-ASCII characters
            phrase    = Regex.Replace(phrase, @"[^\u0000-\u007F]", string.Empty);
            box.Label = ReplaceStartAndEndToken(phrase);
            yield return box;
        }
    }


    public IEnumerable<LabeledPolygon> ParseDescriptionWithPolygonsFromTextAndSpans(
        string     text,
        (int, int) imageSize,
        bool       allowEmptyPhrase  = false,
        string     polygonSepToken   = "<sep>",
        string     polygonStartToken = "<poly>",
        string     polygonEndToken   = "</poly>",
        bool       withBoxAtStart    = false)
    {

        // Ignore <s> </s> and <pad>
        text = text.Replace("<s>", "").Replace("</s>", "").Replace("<pad>", "");

        var phrasePattern = allowEmptyPhrase
            ? $@"(?:(?:<loc_\d+>|{Regex.Escape(polygonSepToken)}|{Regex.Escape(polygonStartToken)}|{Regex.Escape(polygonEndToken)}){{4,}})"
            : $@"([^<]+(?:<loc_\d+>|{Regex.Escape(polygonSepToken)}|{Regex.Escape(polygonStartToken)}|{Regex.Escape(polygonEndToken)}){{4,}})";

        var phrases = Regex.Matches(text, phrasePattern);

        var phraseStringPattern     = @"^\s*(.*?)(?=<od>|</od>|<box>|</box>|<bbox>|</bbox>|<loc_|<poly>)";
        var boxPattern              = $@"((?:<loc_\d+>)+)(?:{Regex.Escape(polygonSepToken)}|$)";
        var polygonsInstancePattern = $@"{Regex.Escape(polygonStartToken)}(.*?){Regex.Escape(polygonEndToken)}";

        foreach (Match phraseText in phrases)
        {
            var box = new LabeledPolygon();

            string phraseTextStrip = Regex.Replace(phraseText.Value, @"^loc_\d+>", "", RegexOptions.None, TimeSpan.FromSeconds(1));

            if (string.IsNullOrEmpty(phraseTextStrip) && !allowEmptyPhrase)
            {
                continue;
            }

            Match phraseMatch = Regex.Match(phraseTextStrip, phraseStringPattern);

            if (!phraseMatch.Success)
            {
                continue;
            }

            string phrase = phraseMatch.Groups[1].Value.Trim();

            IEnumerable<string> polygonsInstancesParsed;

            if (phraseText.Value.Contains(polygonStartToken) && phraseText.Value.Contains(polygonEndToken))
            {
                polygonsInstancesParsed = Regex.Matches(phraseText.Value, polygonsInstancePattern)
                   .Cast<Match>()
                   .Select(m => m.Groups[1].Value);
            }
            else
            {
                polygonsInstancesParsed = new[] { phraseText.Value };
            }

            int index = 0;

            foreach (string polygonsInstance in polygonsInstancesParsed)
            {
                var polygonsParsed = Regex.Matches(polygonsInstance, boxPattern);

                if (polygonsParsed.Count == 0)
                {
                    continue;
                }

                BoundingBox<int>? bbox = null;

                var fullPolygon = new List<Coordinates<float>>();

                foreach (Match polygonParsed in polygonsParsed)
                {
                    var polygon = Regex.Matches(polygonParsed.Groups[1].Value, @"<loc_(\d+)>")
                       .Cast<Match>()
                       .Select(m => int.Parse(m.Groups[1].Value))
                       .ToList();

                    if (withBoxAtStart && bbox is null)
                    {
                        if (polygon.Count > 4)
                        {
                            bbox    = new BoundingBox<int>(polygon.Take(4).ToArray());
                            polygon = polygon.Skip(4).ToList();
                        }
                        else
                        {
                            bbox = new BoundingBox<int>(new int[] { 0, 0, 0, 0 });
                        }
                    }

                    if (polygon.Count % 2 == 1) // abandon last element if is not paired 
                    {
                        polygon = polygon.Take(polygon.Count - 1).ToList();
                    }


                    var dequantizedPolygon = coordinatesQuantizer.Dequantize(
                        polygon.Chunk(2).Select(p => new Coordinates<int>(p[0], p[1])).ToArray(),
                        imageSize
                    );

                    fullPolygon.AddRange(dequantizedPolygon);

                    if (bbox is object)
                    {
                        box.BBoxes.Add(boxQuantizer.Dequantize([bbox], imageSize)[0]);
                    }

                }

                box.Polygon = fullPolygon;
                box.Label   = phrase;
                yield return box;

                index++;
            }
        }
    }

    public string ReplaceStartAndEndToken(string text)
    {
        return text.Replace("<s>", "").Replace("</s>", "");
    }

    public FlorenceResults PostProcessGeneration(string text, TaskTypes task, (int, int) imageSize)
    {
        PostProcessingTypes postProcessingTask = GetPostProcessingType(task);

        switch (postProcessingTask)
        {
            case PostProcessingTypes.pure_text:
            {
                return new FlorenceResults()
                {
                    PureText = ReplaceStartAndEndToken(text)
                };

            }
            case PostProcessingTypes.ocr_with_region:
            {
                var ocrs = ParseOcrFromTextAndSpans(text, imageSize);

                return new FlorenceResults()
                {
                    OCRBBox = ocrs.ToArray()
                };
            }
            case PostProcessingTypes.od:
            case PostProcessingTypes.bboxes:
            case PostProcessingTypes.description_with_bboxes:
            {
                var boxes = ParseDescriptionWithBboxesFromTextAndSpans(
                    text,
                    imageSize: imageSize
                );

                return new FlorenceResults()
                {
                    BoundingBoxes = boxes.ToArray(),
                };
            }
            case PostProcessingTypes.phrase_grounding:
            {
                var bboxes = ParsePhraseGroundingFromTextAndSpans(
                    text,
                    imageSize
                );

                return new FlorenceResults()
                {
                    BoundingBoxes = bboxes.ToArray(),
                };
            }
            case PostProcessingTypes.description_with_polygons:
            {
                var polygons = ParseDescriptionWithPolygonsFromTextAndSpans(text, imageSize);

                return new FlorenceResults()
                {
                    Polygons = polygons.ToArray(),
                };
            }
            case PostProcessingTypes.polygons:
            {
                var polygons = ParseDescriptionWithPolygonsFromTextAndSpans(text, imageSize, true);

                return new FlorenceResults()
                {
                    Polygons = polygons.ToArray(),
                };
            }
            case PostProcessingTypes.description_with_bboxes_or_polygons:
            {
                if (text.Contains("<poly>"))
                {
                    var polygons = ParseDescriptionWithPolygonsFromTextAndSpans(text, imageSize);

                    return new FlorenceResults()
                    {
                        Polygons = polygons.ToArray(),
                    };
                }
                else
                {
                    var bboxes = ParseDescriptionWithBboxesFromTextAndSpans(text, imageSize);

                    return new FlorenceResults()
                    {
                        BoundingBoxes = bboxes.ToArray(),
                    };
                }
            }
            default:
            {
                throw new ArgumentException($"Unknown task answer post processing type: {postProcessingTask}");
            }
        }
    }

}