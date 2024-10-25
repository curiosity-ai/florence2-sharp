using System;
using System.Collections.Generic;

namespace Florence2;

public enum TaskTypes
{
    OCR,
    OCR_WITH_REGION,
    CAPTION,
    DETAILED_CAPTION,
    MORE_DETAILED_CAPTION,
    OD,
    DENSE_REGION_CAPTION,
    CAPTION_TO_PHRASE_GROUNDING,
    REFERRING_EXPRESSION_SEGMENTATION, //TODO pius: not working properly, generated tokens don't seem to be ok
    REGION_TO_SEGMENTATION,
    OPEN_VOCABULARY_DETECTION,
    REGION_TO_CATEGORY,
    REGION_TO_DESCRIPTION,
    REGION_TO_OCR,
    REGION_PROPOSAL
}

public class BoundingBox<T> where T : struct, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
{

    public BoundingBox()
    {
    }
    public BoundingBox(T[] values)
    {
        xmin = values[0];
        ymin = values[1];
        xmax = values[2];
        ymax = values[3];
    }

    public BoundingBox(T xmin, T ymin, T xmax, T ymax)
    {
        this.xmin = xmin;
        this.ymin = ymin;
        this.xmax = xmax;
        this.ymax = ymax;
    }

    public T xmin { get; set; }
    public T ymin { get; set; }
    public T xmax { get; set; }
    public T ymax { get; set; }
}
public class Coordinates<T> where T : struct, IComparable, IComparable<T>, IConvertible, IEquatable<T>, IFormattable
{

    public Coordinates()
    {
    }
    public Coordinates(T[] values)
    {
        x = values[0];
        y = values[1];
    }

    public Coordinates(T x, T y)
    {
        this.x = x;
        this.y = y;
    }

    public T x { get; set; }
    public T y { get; set; }
}
public class LabeledBoundingBoxes
{
    public BoundingBox<float>[] BBoxes { get; set; }
    public string               Label  { get; set; }
}
public class LabeledOCRBox
{
    public Coordinates<float>[] QuadBox { get; set; }
    public string               Text    { get; set; }
}
public class FlorenceResults
{
    public LabeledOCRBox[]        OCRBBox       { get; set; }
    public string                 PureText      { get; set; }
    public LabeledBoundingBoxes[] BoundingBoxes { get; set; }
    public LabeledPolygon[]       Polygons      { get; set; }
}
public class LabeledPolygon
{
    public string                   Label   { get; set; }
    public List<Coordinates<float>> Polygon { get; set; }
    public List<BoundingBox<float>> BBoxes  { get; set; }
}