# User Guide: Deepfake Detection with LLM-Assisted Reasoning

This guide provides detailed instructions on how to use the Deepfake Detection system, from initial setup to interpreting the forensic results.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Basic Usage (CLI)](#basic-usage-cli)
5. [Advanced CLI Options](#advanced-cli-options)
6. [Programmatic API](#programmatic-api)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9 or higher**
- **pip** (Python package installer)
- **Git**

For production runs (using actual LLMs), you will also need:
- An **Azure OpenAI** account with a GPT-4 Vision deployment.

---

## 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/deepfake-detector-llm.git
   cd deepfake-detector-llm
   ```

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Verify installation:**
   ```bash
   deepfake-detector --help
   ```

---

## 3. Configuration

The system uses environment variables for Azure OpenAI authentication. You can set these in a `.env` file in the project root:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt4-vision-deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Alternatively, export them directly in your shell:
```bash
export AZURE_OPENAI_API_KEY="your-key"
```

---

## 4. Basic Usage (CLI)

The primary command is `deepfake-detector detect`.

**Note**: If you haven't installed the package yet, you can run it using `python -m src.deepfake_detector.cli detect ...`.

### Running with a Mock LLM (Recommended for first run)
Use the `mock` backend to verify the pipeline without incurring API costs or requiring credentials.
```bash
# Option A: After installation
deepfake-detector detect \
  --video assets/videos/sample.mp4 \
  --out runs/test_run \
  --llm mock

# Option B: Without installation (from project root)
python -m src.deepfake_detector.cli detect \
  --video assets/videos/sample.mp4 \
  --out runs/test_run \
  --llm mock
```

### Running with Azure OpenAI
```bash
# After installation
deepfake-detector detect \
  --video assets/videos/sample.mp4 \
  --out runs/production_run \
  --llm azure
```

---

## 5. Advanced CLI Options

You can fine-tune the detection process with the following arguments:

- `--num-frames <int>`: (Default: 12) Total frames to extract from the video. More frames provide better temporal coverage but increase processing time.
- `--max-keyframes <int>`: (Default: 8) Maximum number of images to send to the LLM. Fewer keyframes reduce token costs.
- `--llm {mock,azure}`: (Default: mock) Choose the backend.

**Example: High-fidelity analysis**
```bash
deepfake-detector detect --video vid.mp4 --out results/ --num-frames 24 --max-keyframes 12 --llm azure
```

---

## 6. Programmatic API

You can integrate the detection pipeline directly into your Python code:

```python
from deepfake_detector.pipeline import run_pipeline

results = run_pipeline(
    video_path="path/to/video.mp4",
    out_dir="output/analysis",
    llm_backend="azure", # or "mock"
    num_frames=12,
    max_keyframes=8
)

print(f"Decision: {results['label']}")
print(f"Reasoning: {results['reason']}")
```

---

## 7. Understanding the Output

Every run creates a structured directory of artifacts:

```
runs/<run_name>/
├── frames/
│   ├── frame_0000.jpg          # Extracted frames
│   ├── frames_manifest.json    # Metadata for all frames
│   └── evidence_basic.json     # Extracted forensic metrics (motion, blinks, jitter)
├── prompt.txt                  # The exact prompt sent to the LLM
├── llm_output.txt              # The raw response from the LLM
└── decision.json               # The final parsed decision
```

### Interpreting `decision.json`
- **Label: REAL**: No significant forensic anomalies were detected.
- **Label: MANIPULATED**: Evidence (e.g., lack of blinking, unnatural motion) suggests the video is a deepfake.
- **Label: UNCERTAIN**: The evidence was conflicting or insufficient for a clear decision.

---

## 8. Troubleshooting

**Error: "python: can't open file 'deepfake-detector': [Errno 2] No such file or directory"**
- This happens because `deepfake-detector` is a command, not a file. 
- Fix: Ensure you installed the package via `pip install -e .`.
- Alternative: Use `python -m src.deepfake_detector.cli detect ...` instead.

**Error: "Missing Azure OpenAI environment variables"**
- Check your `.env` file or environment variables. Ensure `AZURE_OPENAI_API_KEY` and others are set.

**Error: "Failed to read frame image"**
- Ensure the video file exists and is a valid format (MP4, MOV, AVI).
- Try re-encoding the video with FFmpeg if it seems corrupted.

**Rate Limit Hit**
- The system will automatically retry. If it fails consistently, try reducing `--max-keyframes`.

**Low Accuracy**
- Remember that this is a research tool. Success depends on the specific manipulation type. Refer to `PRD.md` and `ARCHITECTURE.md` for technical limitations.
