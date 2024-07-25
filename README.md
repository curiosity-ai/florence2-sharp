# Florence2

This library implements the Florence-2-base model for advanced image understanding tasks in C#.

## Features

- Image captioning (concise to detailed)
- Optical Character Recognition (OCR)
- Region-based OCR
- Object detection
- Optional phrase grounding


## Quick Start

```csharp
using Florence2;

var modelSource = new FlorenceModelDownloader("./models");
await modelSource.DownloadModelsAsync();
var model = new Florence2Model(modelSource);

using var imgStream       = ...;
var textForPhraseGrounding = ...

var results = modelSession.Run(task, imgStream, textInput: textForPhraseGrounding);
```
