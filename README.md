# ImageToTextTransformers

ImageToTextTransformers is a C# library that implements the Florence-2-base model for advanced image understanding tasks.

## Features

- Image captioning (concise to detailed)
- Optical Character Recognition (OCR)
- Region-based OCR
- Object detection
- Optional phrase grounding


## Quick Start

```csharp
using ImageToTextTransformers;


var modelSource = ...;
await modelSource.InitModelRepo();
var model = new Florence2Model(modelSource);

using var imgStream       = ...;
var textForPhraseGrounding = ...

var results = modelSession.Run(task, imgStream, textInput: textForPhraseGrounding);
```