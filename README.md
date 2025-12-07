# Florence2 — C# Wrapper for Microsoft’s Florence-2 Vision Model

A lightweight, easy-to-use C# library that provides access to Microsoft’s [**Florence-2-base**](https://huggingface.co/microsoft/Florence-2-large) models for advanced image understanding tasks — including captioning, OCR, object detection, and phrase grounding.

This project gives .NET developers a clean API to run Florence-2 locally without needing Python or the original reference implementation.

📦 **NuGet:** [https://www.nuget.org/packages/Florence2](https://www.nuget.org/packages/Florence2)

---

## ✨ Features

* **Image Captioning**
  Generate concise or richly detailed descriptions of images.

* **Optical Character Recognition (OCR)**
  Extract text from entire images or specific regions.

* **Region-based OCR**
  Provide bounding boxes and retrieve text only from selected areas.

* **Object Detection**
  Detect and label objects with bounding boxes.

* **Phrase Grounding (optional)**
  Highlight image regions relevant to a given phrase or textual query.

* **Local Model Execution**
  Automatically downloads and loads the Florence-2-base ONNX models.

---

## 🚀 Quick Start

### 1. Install the package

```bash
dotnet add package Florence2
```

Or get it on NuGet: [https://www.nuget.org/packages/Florence2](https://www.nuget.org/packages/Florence2)

---

### 2. Example Usage

```csharp
using Florence2;

// Download models if needed
var modelSource = new FlorenceModelDownloader("./models");
await modelSource.DownloadModelsAsync();

// Create model instance
var model = new Florence2Model(modelSource);

// Load an image stream
using var imgStream = File.OpenRead("car.jpg");

// Optional text for phrase grounding (may be null)
string phrase = "the red car";

// Choose a task: Captioning / OCR / ObjectDetection / PhraseGrounding / RegionOCR
var task = TaskTypes.OCR_WITH_REGION;

// Run inference
var results = model.Run(task, imgStream, textInput: phrase);

// View results
Console.WriteLine(JsonSerializer.Serialize(results, new JsonSerializerOptions() { WriteIndented = true }));
```

---

## 📚 Supported Tasks

| Task              | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `Caption`         | Generates a caption describing the entire image.      |
| `OCR`             | Extracts all text from the image.                     |
| `RegionOCR`       | Extracts text from specified bounding boxes.          |
| `ObjectDetection` | Detects objects and provides bounding boxes + labels. |
| `PhraseGrounding` | Highlights regions relating to a provided phrase.     |

---

## 📦 Model Files

Models are downloaded automatically via `FlorenceModelDownloader`, but you can also supply your own model directory. The library expects Florence-2-base ONNX models compatible with Microsoft’s open-source release.

---

## 🤝 Contributing

Contributions, issues, and pull requests are welcome! If you find a bug or have a feature request, feel free to open an issue.

---

## 📄 License

MIT — see the `LICENSE` file for details.