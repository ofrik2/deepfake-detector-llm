# Deepfake Detection with LLM-Assisted Evidence Reasoning

[![Tests](https://github.com/ofrik2/deepfake-detector-llm/actions/workflows/tests.yml/badge.svg)](https://github.com/ofrik2/deepfake-detector-llm/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research project exploring whether Large Language Models can assist in deepfake video detection when provided with explicit, interpretable forensic evidence rather than raw video data.

### 🎯 Project Highlights

- **🧪 Research-Oriented**: Focus on interpretability, transparency, and honest failure analysis
- **📦 Packaged as Python Library**: Installable package with CLI and programmatic API
- **✅ 153 Unit Tests**: Comprehensive test coverage with pre-commit hooks
- **🔌 Modular Architecture**: Independent, testable components with clear data contracts
- **💰 Cost-Aware**: Detailed LLM token usage analysis and optimization strategies
- **🔍 Forensic Evidence**: Motion analysis, blink detection, boundary artifacts, face stability
- **🤖 Multiple LLM Backends**: Azure OpenAI for production, mock client for testing
- **📊 Full Transparency**: All artifacts (frames, evidence, prompts, responses) persisted to disk

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Quick Start](#2-quick-start)
3. [Research Motivation](#3-research-motivation)
4. [High-Level System Architecture](#4-high-level-system-architecture)
5. [Repository Structure](#5-repository-structure)
6. [Frame Sampling](#6-frame-sampling)
7. [Evidence Extraction Philosophy](#7-evidence-extraction-philosophy)
8. [Evidence Types](#8-evidence-types)
9. [Blink Detection Experiment](#9-blink-detection-experiment)
10. [Prompt Engineering](#10-prompt-engineering)
11. [LLM Integration](#11-llm-integration)
12. [Experimental Findings](#12-experimental-findings)
13. [Limitations](#13-limitations)
14. [Conclusion](#14-conclusion)
15. [Python Package Structure](#15-python-package-structure)
16. [Testing and Quality Assurance](#16-testing-and-quality-assurance)
17. [How to Run](#17-how-to-run)
18. [Key Features and Capabilities](#18-key-features-and-capabilities)
19. [Development Workflow](#19-development-workflow)
20. [Documentation](#20-documentation)
21. [Final Note](#21-final-note)

---

## 1. Introduction

This project investigates whether Large Language Models (LLMs) can assist in the task of deepfake video detection **when provided with explicit, interpretable visual evidence**, rather than raw video alone.

Traditional deepfake detection approaches rely heavily on:
- large supervised datasets,
- deep convolutional or transformer-based vision models,
- opaque learned decision boundaries.

In contrast, this project explores a complementary research direction:

> Can an LLM reason about *human-interpretable forensic signals* extracted from a video, combine them coherently, and produce a justified decision about authenticity?

The emphasis of this work is **process, experimentation, and failure analysis**, not guaranteed detection success.

---

## 2. Quick Start

Get started with deepfake detection in minutes:

### Installation

```bash
# Clone repository
git clone https://github.com/ofrik2/deepfake-detector-llm.git
cd deepfake-detector-llm

# Install package
pip install -e .

# Verify installation
deepfake-detector --help
```

### Basic Usage (No API Required)

```bash
# Run detection with mock LLM (free, no setup needed)
# If package is installed:
deepfake-detector detect \
  --video path/to/video.mp4 \
  --out results/ \
  --llm mock

# If running from source without installation:
python -m src.deepfake_detector.cli detect \
  --video path/to/video.mp4 \
  --out results/ \
  --llm mock
```

### Production Usage (Azure OpenAI)

```bash
# 1. Create .env file with credentials
cat > .env << EOF
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4-vision
AZURE_OPENAI_API_VERSION=2024-02-15-preview
EOF

# 2. Run detection with Azure OpenAI
deepfake-detector detect \
  --video path/to/video.mp4 \
  --out results/ \
  --llm azure \
  --num-frames 12 \
  --max-keyframes 8
```

### View Results

```bash
# Check decision
cat results/decision.json

# View evidence
cat results/frames/evidence_basic.json | jq

# Read prompt sent to LLM
cat results/prompt.txt
```

### Run Tests

```bash
# Run all 153 tests
pytest tests/ -v

# Quick validation
pytest tests/ -q
```

---

## 3. Research Motivation

The motivation for this project is threefold:

1. **Interpretability**
   - Most deepfake detectors provide little explanation for their decisions.
   - This project forces every signal to be explicit and explainable.

2. **LLM Reasoning Abilities**
   - Modern LLMs excel at combining weak, heterogeneous evidence.
   - They can reason, hedge, and express uncertainty.

3. **Scientific Curiosity**
   - The professor explicitly questioned whether success was likely.
   - The project therefore focuses on exploring feasibility and limitations.

A core assumption throughout this work is:

> Failure to detect manipulation reliably is itself a meaningful and reportable result.

---

## 4. High-Level System Architecture

The system is designed as a **modular pipeline**, where each stage is independently testable and explainable.

```
Video
  ↓
Frame Sampling
  ↓
Evidence Extraction (Interpretable Signals)
  ↓
Prompt Construction
  ↓
LLM Reasoning
  ↓
Decision (REAL / MANIPULATED / UNCERTAIN)
```

Each stage emits artifacts saved to disk, allowing full inspection of intermediate results.

---

## 5. Repository Structure

```
deepfake-detector-llm/
├── assets/
│   └── videos/                 # Input videos
│
├── src/deepfake_detector/
│   ├── cli.py                  # Command-line interface
│   ├── pipeline.py             # End-to-end orchestration
│   ├── config.py               # Configuration settings
│   ├── types.py                # Shared type definitions
│   │
│   ├── video/
│   │   ├── reader.py           # Video loading utilities
│   │   ├── frames.py           # Frame sampling + manifest creation
│   │   └── quality_checks.py   # Video quality validation
│   │
│   ├── evidence/
│   │   ├── basic_signals.py    # Core forensic evidence extraction
│   │   └── blink.py            # Low-level blink feature extraction
│   │
│   ├── prompts/
│   │   ├── prompt_builder.py   # Converts evidence into LLM prompt
│   │   └── prompt_log.md       # Prompt iteration history
│   │
│   ├── llm/
│   │   ├── azure_client.py     # Azure OpenAI integration
│   │   ├── mock_client.py      # Deterministic mock LLM
│   │   └── client_base.py      # Base class for LLM clients
│   │
│   ├── llm_input/
│   │   └── input_builder.py    # Builds structured LLM input
│   │
│   └── decision/
│       └── parser.py           # Parses LLM output into structured decision
│
├── runs/
│   └── <run_name>/             # Per-run outputs
│       ├── frames/
│       │   ├── frames_manifest.json
│       │   └── evidence_basic.json
│       ├── llm_input.json
│       ├── prompt.txt
│       ├── llm_output.txt
│       └── decision.json
│
├── tests/                      # Unit tests for all components
│   ├── test_cli.py             # Tests for CLI functionality
│   ├── test_pipeline.py        # Tests for pipeline orchestration
│   ├── test_llm_clients.py     # Tests for LLM client implementations
│   ├── test_prompt_builder.py  # Tests for prompt construction
│   ├── test_video_*.py         # Tests for video processing
│   ├── test_evidence_*.py      # Tests for evidence extraction
│   └── ...                     # Additional test files
│
├── docs/
│   ├── ARCHITECTURE.md         # System architecture documentation
│   ├── PRD.md                  # Product requirements document
│   └── USER_GUIDE.md           # User guide and examples
│
├── scripts/
│   ├── export_report.py        # Report generation utilities
│   └── run_local.sh            # Local execution script
│
├── pyproject.toml              # Python package configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 6. Frame Sampling

To control computational cost and simplify reasoning, videos are sampled uniformly.

### Manifest Contents
Each run produces a `frames_manifest.json` containing:
- frame file paths
- timestamps (seconds)
- frames-per-second (FPS)
- video duration
- sampling mode

This manifest acts as the contract between video processing and evidence extraction.

---

## 7. Evidence Extraction Philosophy

All evidence is designed to be:
- deterministic,
- explainable,
- cheap to compute,
- human-interpretable.

No learned models are used at this stage.

The goal is *not* optimal detection performance, but **forensic clarity**.

---

## 8. Evidence Types

### 8.1 Motion-Based Evidence
- Global frame-to-frame motion
- Mouth-region motion
- Eye-region motion
- Eyes-over-mouth motion ratio

Used to detect unnatural motion decoupling (e.g., speaking mouth with static eyes).

### 8.2 Face Stability Evidence
- Face detection missing ratio
- Bounding box center jitter (normalized)
- Bounding box size jitter (normalized)

High instability may indicate compositing or tracking failure.

### 8.3 Blink-Related Evidence
- Eye openness proxy over time
- Blink-like event count (strict heuristic)
- Blink dip fraction
- Eye openness range

A specific experiment focused on detecting **absence of blinking**, a known deepfake artifact.

### 8.4 Boundary / Compositing Evidence
- Edge strength at face boundary
- Edge strength in background region
- Log-ratio calibration (face vs background)

Designed to detect mask-edge artifacts without requiring segmentation.

---

## 9. Blink Detection Experiment

A dedicated experiment was conducted on videos where:
- the subject speaks,
- eyes do not blink.

Despite human observers often suspecting manipulation, automated detection proved difficult.

### Challenges Identified
- ROI drift causes noisy openness signals
- Lighting and compression introduce false dips
- Sparse frame sampling hides temporal events

Multiple iterations were required:
- ROI resizing
- Log-scaled openness metrics
- Strict blink event definitions
- Increased frame counts

Even then, results often remained **UNCERTAIN**.

---

## 10. Prompt Engineering

Evidence is converted into a structured prompt containing:
- context (frames, duration, FPS),
- explicit numeric evidence,
- heuristic interpretation guidance,
- strict output format requirements.

The prompt is saved verbatim to `prompt.txt` for transparency.

Prompt engineering proved critical:
- unclear metrics caused hallucinated reasoning,
- ratios without calibration were misinterpreted,
- explicit heuristic explanations improved consistency.

---

## 11. LLM Integration

The system supports:
- Azure OpenAI (vision + text)
- Mock deterministic LLM (for testing)

The LLM is **not trained** — it is only asked to reason.

The model outputs:
```
Label: REAL | MANIPULATED | UNCERTAIN
Reason: <short justification>
```

UNCERTAIN is treated as a valid and meaningful outcome.

### Cost Analysis and Estimation

Using Azure OpenAI incurs costs based on token usage for both input (prompt) and output (response). Understanding and optimizing these costs is critical for production use.

#### Token Usage Breakdown

**Per-Video Analysis Costs:**
- **Base Prompt**: ~800-1200 tokens (evidence descriptions, instructions, context)
- **Per-Frame Image**: ~500-1000 tokens (varies with image resolution and content)
- **LLM Response**: ~50-200 tokens (label + reasoning)

**Example Cost Calculation** (assuming GPT-4 Vision pricing):
```
Input tokens per run:
  - Base prompt:        1,000 tokens
  - Images (8 frames):  8 × 750 = 6,000 tokens
  - Total input:        7,000 tokens

Output tokens:          100 tokens

Cost per run (estimated):
  - Input:  7,000 tokens × $0.01/1K  = $0.07
  - Output: 100 tokens   × $0.03/1K  = $0.003
  - Total per video:                  ≈ $0.073
```

#### Cost Optimization Strategies

1. **Frame Sampling Control**
   - Use `--num-frames` to control initial frame extraction (default: 12)
   - Use `--max-keyframes` to limit frames sent to LLM (default: 8)
   - Reducing keyframes from 8 to 4 can cut image token costs by ~50%

2. **Development Mode**
   - Use `--llm mock` during development to avoid API costs entirely
   - Mock client provides deterministic responses for testing
   - Switch to `--llm azure` only for production/validation runs

3. **Batch Processing**
   - Process multiple videos in sequence to amortize setup costs
   - Monitor usage via Azure OpenAI portal to track spending

4. **Image Optimization**
   - Frames are automatically resized (default `resize_max=512`)
   - Smaller images reduce token usage without significant quality loss
   - Lower resolution suitable for motion/blink analysis

#### Monitoring Usage

The Azure OpenAI client captures usage information:
```python
# Usage data is stored in LLMResponse.usage
{
  "prompt_tokens": 7000,
  "completion_tokens": 100,
  "total_tokens": 7100
}
```

This data is logged and can be aggregated across runs for cost tracking and analysis.

#### Monthly Cost Estimates

Assuming 100 videos per month with default settings:
- Cost per video: ~$0.07
- Monthly total: ~$7.00

For research/academic use, Azure offers free tiers and student credits that may cover experimental workloads.

---

## 12. Experimental Findings

### Key Observations
1. Stable heuristics often bias toward REAL.
2. Blink absence is surprisingly hard to detect reliably.
3. Boundary evidence is easily misinterpreted.
4. LLMs reflect the quality of evidence provided.

### Important Result
> Even with explicit evidence, an LLM may reasonably conclude UNCERTAIN.

This aligns with the professor’s original skepticism.

---

## 13. Limitations

- No facial landmarks or EAR metrics
- No optical flow
- No audio-visual alignment
- Limited frame sampling
- Heuristics are brittle by nature

These limitations are explicitly documented rather than hidden.

---

## 14. Conclusion

This project demonstrates:
- a principled attempt at LLM-assisted deepfake detection,
- a transparent, inspectable pipeline,
- honest analysis of failure modes.

The inability to reliably detect manipulation is itself a valuable conclusion.

---

## 15. Python Package Structure

This project is packaged as a proper Python package using modern packaging standards, making it easy to install, distribute, and use as a library or command-line tool.

### Package Configuration

The package is configured via `pyproject.toml` following [PEP 518](https://www.python.org/dev/peps/pep-0518/) standards:

**Package Metadata:**
- **Name**: `deepfake-detector-llm`
- **Version**: 0.1.0
- **Python Requirement**: >=3.9
- **License**: MIT

**Core Dependencies:**
```
numpy>=1.23          # Numerical operations
opencv-python>=4.8   # Video/image processing
Pillow>=9.5          # Image handling
python-dotenv>=1.0   # Environment configuration
tqdm>=4.66           # Progress bars
openai>=1.0.0        # Azure OpenAI API client
```

**Development Dependencies:**
```
pytest>=7.4          # Testing framework
black>=23.0          # Code formatting
isort>=5.12          # Import sorting
mypy>=1.5            # Static type checking
```

### Installation Methods

**Development Mode (Editable):**
```bash
# Install in editable mode with dev dependencies
pip install -e .

# Or install with dev extras
pip install -e ".[dev]"
```

**Production Installation:**
```bash
# Install from source
pip install .

# Or from a wheel distribution
pip install deepfake-detector-llm-0.1.0-py3-none-any.whl
```

**From Git Repository:**
```bash
pip install git+https://github.com/ofrik2/deepfake-detector-llm.git
```

### Command-Line Interface

The package registers a console entry point via `pyproject.toml`:

```toml
[project.scripts]
deepfake-detector = "src.deepfake_detector.cli:main"
```

After installation, the `deepfake-detector` command becomes globally available:

```bash
# Run detection
deepfake-detector detect --video input.mp4 --out results/ --llm azure

# Get help
deepfake-detector --help
deepfake-detector detect --help
```

### Package Structure

```
deepfake-detector-llm/
├── pyproject.toml              # Modern package configuration
├── requirements.txt            # Direct dependencies (for pip install -r)
├── setup.py                    # (Optional) legacy support
│
└── src/
    └── deepfake_detector/      # Main package namespace
        ├── __init__.py
        ├── cli.py              # CLI entry point
        ├── pipeline.py         # Main API
        │
        └── [subpackages]/      # Feature modules
```

### Programmatic API

Besides the CLI, the package can be used as a library:

```python
from deepfake_detector.pipeline import run_pipeline

# Run detection programmatically
results = run_pipeline(
    video_path="video.mp4",
    out_dir="output/",
    llm_backend="azure",
    num_frames=12,
    max_keyframes=8
)

print(f"Detection result: {results['label']}")
print(f"Output saved to: {results['decision_path']}")
```

### Build and Distribution

**Build wheel package:**
```bash
python -m build
```

This creates:
- `dist/deepfake_detector_llm-0.1.0-py3-none-any.whl`
- `dist/deepfake-detector-llm-0.1.0.tar.gz`

**Upload to PyPI** (if desired):
```bash
python -m twine upload dist/*
```

---

## 16. Testing and Quality Assurance

This project maintains high code quality through comprehensive unit testing and automated checks, ensuring reliability and preventing regressions.

### Unit Test Coverage

The codebase includes **153 unit tests** providing extensive coverage across all major components:

#### Test Breakdown by Module:

**1. Video Processing (`test_video_*.py`)**
- Frame extraction with uniform and every-N sampling strategies
- Aspect-ratio preserving resize operations
- Video metadata reading (FPS, frame count, duration)
- Quality checks (brightness, blur detection using Laplacian variance)
- Edge cases: empty videos, corrupt files, missing frames
- Frame manifest creation and serialization

**2. Evidence Computation (`test_evidence_*.py`)**
- Basic signals extraction:
  - Motion analysis (global, mouth region, eye region)
  - Face stability metrics (detection failures, bounding box jitter)
  - Boundary/compositing evidence (edge strength ratios)
- Blink detection algorithms:
  - Eye openness proxy computation
  - Blink-like event detection with strict heuristics
  - Temporal analysis and noise handling
  - ROI (Region of Interest) tracking and drift compensation
- Edge cases: no faces detected, single-frame videos, invalid ROIs
- Evidence serialization and deserialization

**3. LLM Integration (`test_llm_clients.py`)**
- Mock client deterministic behavior
- Azure OpenAI client initialization and authentication
- Prompt + image input handling
- Response parsing and usage tracking
- Rate limiting and retry logic
- Error handling for API failures

**4. LLM Input Building (`test_llm_input_builder.py`)**
- Keyframe selection strategies
- Evidence aggregation from manifest + evidence files
- Structured input format generation
- Temporal sampling and frame metadata inclusion
- JSON serialization consistency

**5. Prompt Engineering (`test_prompt_builder.py`)**
- Evidence-to-text conversion
- Heuristic interpretation guidance inclusion
- Output format specification
- Context information embedding (FPS, duration, sampling mode)
- Template consistency across different evidence types

**6. Decision Parsing (`test_decision_parser.py`)**
- LLM output interpretation with various formats:
  - Standard format: `Label: REAL/MANIPULATED/UNCERTAIN`
  - Malformed outputs (missing fields, incorrect labels)
  - Extra whitespace and case variations
- Reason extraction and validation
- Error handling for unparseable outputs

**7. Pipeline Orchestration (`test_pipeline.py`)**
- End-to-end flow with mocked dependencies
- Artifact generation and persistence
- Output directory creation
- Integration between components
- Error propagation and handling

**8. CLI Interface (`test_cli.py`)**
- Argument parsing for all commands
- Parameter validation
- Help text generation
- Command execution with mocked pipeline

### Testing Infrastructure

**Framework**: `pytest` with `pytest-mock`

**Key Testing Strategies:**
- **Comprehensive mocking**: External dependencies (OpenCV, file I/O, LLM APIs) are mocked to ensure:
  - Fast execution (no actual video processing or API calls)
  - Deterministic results
  - Isolation of units under test
  
- **Edge case coverage**: Tests explicitly cover:
  - Empty inputs
  - Boundary conditions (single frame, maximum frames)
  - Error conditions (missing files, API failures)
  - Invalid data formats
  
- **Fixtures**: Reusable test fixtures provide consistent test data:
  - Mock video frames (various sizes, colors)
  - Mock manifests with different sampling strategies
  - Mock evidence with typical and edge-case values

### Running Tests

To run the full test suite locally, ensure you have installed the development dependencies:

```bash
pip install -e ".[dev]"
```

Then execute pytest from the project root:

```bash
pytest
```

For a quick run (quiet mode):

```bash
pytest -q
```

To check code coverage:

```bash
pytest --cov=src/deepfake_detector --cov-report=term-missing
```

### Continuous Integration (CI)

This repository uses GitHub Actions for continuous integration. The [Tests workflow](.github/workflows/tests.yml) automatically runs on every push and pull request to `main` and `master` branches.

The CI pipeline performs the following steps:
1.  **Test Job**: Runs the full test suite on Python 3.10 and 3.11.
2.  **Lint Job**: Checks code formatting and quality using `ruff`, `black`, and `isort`.

---

## 17. How to Run

### Prerequisites

**System Requirements:**
- Python 3.9 or higher
- pip package manager
- Git (for cloning repository)

**Azure OpenAI Account** (for production use):
- Active Azure subscription
- Azure OpenAI resource provisioned
- GPT-4 Vision deployment created

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/ofrik2/deepfake-detector-llm.git
cd deepfake-detector-llm
```

**2. Install dependencies:**
```bash
# Install in development mode
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

**3. Verify installation:**
```bash
deepfake-detector --help
```

### Environment Setup

Create a `.env` file in the project root with Azure OpenAI credentials:

```bash
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT_NAME=your-gpt4-vision-deployment
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**Alternative**: Export as environment variables:
```bash
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
export AZURE_OPENAI_API_KEY=your-api-key-here
export AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
export AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Usage Examples

**Basic Detection (using mock LLM):**
```bash
deepfake-detector detect \
  --video assets/videos/example.mp4 \
  --out runs/example \
  --llm mock \
  --num-frames 12 \
  --max-keyframes 8
```

**Production Detection (using Azure OpenAI):**
```bash
deepfake-detector detect \
  --video assets/videos/suspicious.mp4 \
  --out runs/suspicious_analysis \
  --llm azure \
  --num-frames 20 \
  --max-keyframes 10
```

**Quick Test with Fewer Frames:**
```bash
deepfake-detector detect \
  --video test.mp4 \
  --out runs/quick_test \
  --llm mock \
  --num-frames 6 \
  --max-keyframes 4
```

### Command-Line Arguments

**Required Arguments:**
- `--video`: Path to input video file (mp4, avi, mov, etc.)
- `--out`: Output directory for artifacts and results

**Optional Arguments:**
- `--llm`: LLM backend to use (choices: `mock`, `azure`; default: `mock`)
- `--num-frames`: Number of frames to extract from video (default: `12`)
- `--max-keyframes`: Maximum keyframes to send to LLM (default: `8`)

**Notes:**
- `--num-frames`: Controls initial frame extraction (more frames = better temporal coverage)
- `--max-keyframes`: Controls token cost (fewer keyframes = lower API cost)
- `--llm mock`: Free, deterministic, useful for development/testing
- `--llm azure`: Production mode, requires valid Azure credentials

### Output Structure

After running detection, the output directory contains:

```
runs/<run_name>/
├── frames/
│   ├── frame_0000.jpg              # Extracted frame images
│   ├── frame_0001.jpg
│   ├── ...
│   ├── frames_manifest.json        # Frame metadata (timestamps, FPS, etc.)
│   └── evidence_basic.json         # Computed forensic evidence
│
├── llm_input.json                  # Structured LLM input
├── prompt.txt                      # Complete prompt sent to LLM
├── llm_output.txt                  # Raw LLM response
└── decision.json                   # Final parsed decision
```

### Interpreting Results

**decision.json format:**
```json
{
  "label": "MANIPULATED",
  "reason": "The video exhibits unnaturally low blinking frequency (0 blink events detected across 12 frames) combined with high eye-over-mouth motion decoupling (ratio 0.15), suggesting facial manipulation or replacement.",
  "raw_text": "Label: MANIPULATED\nReason: ..."
}
```

**Possible Labels:**
- `REAL`: Evidence suggests authentic video
- `MANIPULATED`: Evidence suggests AI generation or manipulation
- `UNCERTAIN`: Insufficient or conflicting evidence for confident classification

**Viewing Evidence:**

Examine `evidence_basic.json` to see all computed metrics:
```json
{
  "global_motion_mean": 8.45,
  "mouth_motion_mean": 12.3,
  "eyes_motion_mean": 1.8,
  "eyes_over_mouth_motion_ratio": 0.146,
  "face_bbox_none_ratio": 0.0,
  "blink_like_events": 0,
  "blink_dip_fraction": 0.02,
  ...
}
```

### Programmatic Usage

Use as a Python library in your own scripts:

```python
from deepfake_detector.pipeline import run_pipeline

# Run detection
results = run_pipeline(
    video_path="path/to/video.mp4",
    out_dir="output/analysis",
    llm_backend="azure",  # or "mock"
    num_frames=12,
    max_keyframes=8
)

# Access results
print(f"Detection: {results['label']}")
print(f"Artifacts saved in: {results['out_dir']}")
print(f"Decision file: {results['decision_path']}")

# Read parsed decision
import json
with open(results['decision_path']) as f:
    decision = json.load(f)
    print(f"Reason: {decision['reason']}")
```

### Troubleshooting

**"Missing Azure OpenAI environment variables"**
- Ensure `.env` file exists with all required variables
- Or export variables to shell environment
- Use `--llm mock` if you don't have Azure credentials

**"Failed to read frame image"**
- Check video file is not corrupted
- Verify video codec is supported by OpenCV
- Try re-encoding video: `ffmpeg -i input.mp4 -c:v libx264 output.mp4`

**"Rate limit hit"**
- Azure OpenAI client automatically retries with backoff
- Reduce `--max-keyframes` to send fewer images
- Wait 60 seconds and retry

**Tests fail on commit**
- Pre-commit hook prevents broken commits
- Run `pytest tests/ -v` to see detailed failures
- Fix issues before committing
- Override with `git commit --no-verify` (not recommended)

---

## 18. Key Features and Capabilities

This project implements a comprehensive deepfake detection system with numerous features designed for transparency, reproducibility, and research flexibility.

### Core Capabilities

**1. Modular Plugin Architecture**
- **Extensible Detectors**: Add new detection logic without modifying the core pipeline.
- **Dynamic Discovery**: Detectors are automatically discovered from `src/deepfake_detector/detectors/`, a `plugins/` folder at the repo root, or via the `DEEPFAKE_DETECTOR_PLUGINS_PATH` environment variable.
- **CLI Integration**: Select detectors via `--detector <name>` and list available ones with `--list-detectors`.
- Artifacts saved at each stage for inspection and debugging.

**2. Multiple LLM Backend Support**
- **Azure OpenAI**: Production-ready vision model integration
- **Mock Client**: Deterministic responses for testing without API costs
- Extensible architecture allows adding new providers (Gemini, Anthropic, etc.)

**3. Flexible Frame Sampling**
- **Uniform sampling**: Evenly distributed frames across video duration
- **Every-N sampling**: Take every Nth frame (useful for high FPS videos)
- Configurable frame count and keyframe selection
- Automatic aspect-ratio preserving resize

**4. Comprehensive Evidence Extraction**

**Motion Analysis:**
- Global frame-to-frame motion detection
- Region-specific motion (mouth, eyes)
- Motion ratio metrics (eyes/mouth) to detect decoupling

**Face Stability:**
- Face detection consistency tracking
- Bounding box jitter analysis (position and size)
- Face presence ratio across frames

**Blink Detection:**
- Eye openness proxy computation
- Blink-like event detection with configurable thresholds
- Temporal analysis for blink frequency patterns
- Handles ROI drift and lighting variations

**Boundary/Compositing Evidence:**
- Edge strength analysis at face boundaries
- Calibrated face-vs-background edge ratios
- Log-scaled metrics to detect mask artifacts

**5. Prompt Engineering Framework**
- Evidence-to-text conversion with heuristic guidance
- Structured output format enforcement
- Context-rich prompts with temporal information
- Interpretable evidence descriptions for LLM reasoning

**6. Quality Assurance**
- Brightness validation to detect overly dark frames
- Blur detection using Laplacian variance
- Video metadata validation (FPS, frame count, duration)
- Comprehensive error handling and reporting

**7. Artifact Persistence**
- All intermediate results saved to disk
- Frame manifest with precise timestamps
- JSON-formatted evidence for programmatic access
- Complete prompt visibility for reproducibility
- Raw LLM output preserved alongside parsed decision

**8. Programmatic API**
- Use as library in custom scripts
- Direct access to pipeline components
- Structured return values with all artifact paths
- No CLI required for integration

### Advanced Features

**Rate Limiting and Retry Logic**
- Automatic retry on Azure OpenAI rate limits
- Exponential backoff with configurable wait times
- Graceful error handling for transient failures

**Image Encoding for Vision Models**
- Automatic base64 encoding for Azure OpenAI
- Data URL generation for local image files
- Efficient batch image processing

**Deterministic Testing**
- Mock LLM client extracts evidence from prompt
- Rule-based label assignment for validation
- No external API dependencies in tests

**Cost Optimization**
- Frame count control to manage extraction overhead
- Keyframe limiting to reduce token usage
- Image resize to minimize payload size
- Mock mode for unlimited cost-free experimentation

**Type Safety**
- Comprehensive type hints throughout codebase
- Protocol-based LLM client interface
- Dataclass-based structured data
- mypy validation for static type checking

**Cross-Platform Compatibility**
- Works on Windows, Linux, macOS
- Standard Python packaging (no platform-specific builds)
- Relative path handling for portability

### Research-Oriented Design

**Transparency:**
- Every decision is traceable to evidence
- Prompts are human-readable and editable
- LLM reasoning is explicitly captured
- Failure modes are documented, not hidden

**Reproducibility:**
- Deterministic sampling strategies
- Fixed random seeds where applicable
- Version-controlled prompt templates
- Artifact persistence enables replay

**Extensibility:**
- Add new evidence extractors by implementing standard interface
- Plug in new LLM providers via Protocol pattern
- Custom quality checks can be inserted
- Prompt templates are separate from logic

**Educational Value:**
- Clear code structure for learning
- Extensive documentation and comments
- Unit tests serve as usage examples
- Architecture docs explain design decisions

### Limitations and Future Work

**Current Limitations:**
- No facial landmark detection (no EAR metrics)
- No optical flow analysis
- No audio-visual synchronization checks
- Limited to single-face videos
- No temporal transformer models

**Potential Enhancements:**
- Add MediaPipe for landmark-based features
- Implement dense optical flow analysis
- Audio deepfake detection integration
- Multi-face tracking and analysis
- Ensemble LLM voting for robustness
- Active learning for prompt optimization
- Real-time processing pipeline
- Web UI for interactive exploration

---

## 19. Development Workflow

### Setting Up Development Environment

**1. Clone and Install:**
```bash
git clone https://github.com/ofrik2/deepfake-detector-llm.git
cd deepfake-detector-llm
pip install -e ".[dev]"
```

**2. Verify Setup:**
```bash
# Run tests
pytest tests/ -v

# Check code formatting
black --check src/ tests/

# Verify imports
isort --check src/ tests/

# Run type checker
mypy src/
```

### Development Cycle

**1. Make Changes:**
- Implement features in `src/deepfake_detector/`
- Follow existing code structure and patterns
- Add type hints to all functions

**2. Write Tests:**
- Create tests in `tests/` matching module structure
- Aim for >90% coverage of new code
- Test edge cases and error conditions

**3. Validate:**
```bash
# Run tests
pytest tests/ -v

# Format code
black src/ tests/
isort src/ tests/

# Check types
mypy src/
```

**4. Commit:**
```bash
git add .
git commit -m "Description of changes"
# Pre-commit hook runs tests automatically
```

### Code Style Guidelines

**Formatting:**
- Line length: 100 characters (Black default)
- Use double quotes for strings
- Sort imports with isort (Black-compatible profile)

**Naming Conventions:**
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`
- Private functions: `_leading_underscore`

**Type Hints:**
```python
def process_frame(
    frame: np.ndarray,
    resize_max: int = 512
) -> Dict[str, Any]:
    ...
```

**Documentation:**
- Docstrings for all public functions and classes
- Inline comments for complex logic
- Update README for user-facing changes

### Adding a New Detector Plugin

**1. Create the Detector Class:**
Create a new Python file (e.g., `my_detector.py`) in `src/deepfake_detector/detectors/` or a `plugins/` folder at the repo root.

```python
from src.deepfake_detector.detectors.base import BaseDetector, DetectorResult
from src.deepfake_detector.detectors.registry import register_detector

@register_detector("my-custom-detector")
class MyCustomDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "my-custom-detector"

    def detect(self, video_path, out_dir, config=None) -> DetectorResult:
        # Implementation logic here
        return DetectorResult(
            label="REAL",
            rationale="Explanation of the decision",
            evidence_used=["path/to/evidence.json"]
        )
```

**2. Use from CLI:**
```bash
deepfake-detector detect --video video.mp4 --out results/ --detector my-custom-detector
```

### Adding New Evidence Types

**1. Create Extractor Module:**
```python
# src/deepfake_detector/evidence/new_evidence.py

def compute_new_evidence(manifest_path: str) -> Dict[str, Any]:
    """
    Extract new evidence type from video frames.
    
    Args:
        manifest_path: Path to frames manifest JSON
        
    Returns:
        Dictionary with evidence metrics
    """
    # Implementation here
    return {
        "metric_1": value1,
        "metric_2": value2,
    }
```

**2. Integrate into Pipeline:**
```python
# Update src/deepfake_detector/pipeline.py
from .evidence.new_evidence import compute_new_evidence

def run_pipeline(...):
    ...
    new_evidence = compute_new_evidence(str(manifest_path))
    # Merge into evidence dict
```

**3. Update Prompt Builder:**
```python
# Update src/deepfake_detector/prompts/prompt_builder.py
def build_prompt_text(llm_input):
    ...
    if "metric_1" in evidence:
        lines.append(f"- New metric: {evidence['metric_1']}")
```

**4. Write Tests:**
```python
# tests/test_evidence_new.py
def test_compute_new_evidence():
    result = compute_new_evidence("path/to/manifest.json")
    assert "metric_1" in result
    assert isinstance(result["metric_1"], float)
```

### Adding New LLM Providers

**1. Implement Client:**
```python
# src/deepfake_detector/llm/new_provider_client.py
from dataclasses import dataclass
from .client_base import LLMClient, LLMResponse

@dataclass
class NewProviderClient(LLMClient):
    def generate(self, *, prompt: str, image_paths=None) -> LLMResponse:
        # Call provider API
        response = call_api(prompt, image_paths)
        return LLMResponse(
            raw_text=response.text,
            model_name="new-provider",
            usage=response.usage
        )
```

**2. Register in CLI:**
```python
# src/deepfake_detector/cli.py
detect.add_argument(
    "--llm",
    default="mock",
    choices=["mock", "azure", "new-provider"],
    help="LLM backend"
)
```

**3. Update Pipeline:**
```python
# src/deepfake_detector/pipeline.py
elif llm_backend == "new-provider":
    from .llm.new_provider_client import NewProviderClient
    client = NewProviderClient()
```

### Debugging Tips

**View Pipeline Artifacts:**
```bash
# After running detection
ls -R runs/example/

# Inspect evidence
cat runs/example/frames/evidence_basic.json | jq

# Read prompt
cat runs/example/prompt.txt

# Check LLM response
cat runs/example/llm_output.txt
```

**Test Individual Components:**
```python
# Test frame extraction alone
from deepfake_detector.video.frames import extract_frames

manifest = extract_frames(
    "video.mp4",
    "output/frames",
    mode="uniform",
    num_frames=12
)
print(manifest)
```

**Use Mock Mode for Rapid Iteration:**
```bash
# No API costs, instant responses
deepfake-detector detect \
  --video test.mp4 \
  --out runs/debug \
  --llm mock
```

---

## 20. Documentation

Comprehensive documentation is provided in the `docs/` directory:

**[ARCHITECTURE.md](docs/ARCHITECTURE.md)**
- System design and component interactions
- Data flow diagrams
- Interface specifications
- Design decisions and rationale

**[PRD.md](docs/PRD.md)**
- Product requirements document
- Project goals and success criteria
- Target users and use cases
- Non-goals and scope boundaries

**[USER_GUIDE.md](docs/USER_GUIDE.md)**
- Step-by-step tutorials
- Common workflows
- Example use cases
- FAQ and troubleshooting

### Additional Resources

**Prompt Iteration Log:**
- [prompts/prompt_log.md](prompt_log.md)
- Documents prompt engineering iterations
- Shows evolution of prompts with reasoning
- Useful for understanding LLM interaction design

**Test Files as Examples:**
- Tests demonstrate API usage patterns
- Show expected inputs and outputs
- Serve as living documentation

---

## 21. Final Note

This work prioritizes **scientific honesty over success metrics**.

From an academic standpoint, the project succeeds by:
- exploring a difficult question,
- iterating thoughtfully,
- documenting limitations clearly.

made with ❤️ for the academic learning of LLMs in multi-agent environments.