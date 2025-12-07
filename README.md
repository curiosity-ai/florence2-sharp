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

| Task | Description |
| :--- | :--- |
| `TaskTypes.OCR` | **Optical Character Recognition**: Extracts all text recognized in the image. |
| `TaskTypes.OCR_WITH_REGION` | Extracts all text from the image and provides the bounding box (quad-box) for each detected text region. |
| `TaskTypes.CAPTION` | Generates a brief caption describing the entire image. |
| `TaskTypes.DETAILED_CAPTION` | Generates a detailed description of the image, covering more elements than the standard caption. |
| `TaskTypes.MORE_DETAILED_CAPTION` | Generates a highly comprehensive and lengthy description of the image contents. |
| `TaskTypes.OD` | **Object Detection**: Detects objects in the image and provides their bounding boxes and class labels. |
| `TaskTypes.DENSE_REGION_CAPTION` | Detects a large number of regions (densely packed) and provides a caption/label for each bounding box. |
| `TaskTypes.CAPTION_TO_PHRASE_GROUNDING`| **Phrase Grounding**: Highlights/localizes regions (bounding boxes) that correspond to specific phrases provided in a text input. |
| `TaskTypes.REGION_TO_SEGMENTATION` | Generates a segmentation mask for an object defined by a provided bounding box. |
| `TaskTypes.OPEN_VOCABULARY_DETECTION` | Detects objects matching a provided text prompt (similar to phrase grounding, but often used to detect specific classes). |
| `TaskTypes.REGION_TO_CATEGORY` | Classifies the object contained within a specific provided bounding box. |
| `TaskTypes.REGION_TO_DESCRIPTION` | Generates a description or caption for a specific region defined by a provided bounding box. |
| `TaskTypes.REGION_TO_OCR` | Extracts text specifically from a region defined by a provided bounding box. |
| `TaskTypes.REGION_PROPOSAL` | Identifies and outputs bounding boxes for salient regions or potential objects in the image without labels. |

---

## 📦 Model Files

Models are downloaded automatically via `FlorenceModelDownloader`, but you can also supply your own model directory. The library expects Florence-2-base ONNX models compatible with Microsoft’s open-source release.

---

## 🤝 Contributing

Contributions, issues, and pull requests are welcome! If you find a bug or have a feature request, feel free to open an issue.

---

## 📄 License

MIT — see the `LICENSE` file for details.