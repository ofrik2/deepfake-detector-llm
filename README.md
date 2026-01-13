# Deepfake Detection with LLM-Assisted Evidence Reasoning

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

## 2. Research Motivation

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

## 3. High-Level System Architecture

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

## 4. Repository Structure

```
deepfake-detector-llm/
├── assets/
│   └── videos/                 # Input videos
│
├── src/deepfake_detector/
│   ├── cli.py                  # Command-line interface
│   ├── pipeline.py             # End-to-end orchestration
│   │
│   ├── video/
│   │   ├── reader.py           # Video loading utilities
│   │   └── frames.py           # Frame sampling + manifest creation
│   │
│   ├── evidence/
│   │   ├── basic_signals.py    # Core forensic evidence extraction
│   │   └── blink.py            # Low-level blink feature extraction
│   │
│   ├── prompts/
│   │   └── prompt_builder.py   # Converts evidence into LLM prompt
│   │
│   └── llm/
│       ├── azure_client.py     # Azure OpenAI integration
│       └── mock_client.py      # Deterministic mock LLM
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
└── README.md
```

---

## 5. Frame Sampling

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

## 6. Evidence Extraction Philosophy

All evidence is designed to be:
- deterministic,
- explainable,
- cheap to compute,
- human-interpretable.

No learned models are used at this stage.

The goal is *not* optimal detection performance, but **forensic clarity**.

---

## 7. Evidence Types

### 7.1 Motion-Based Evidence
- Global frame-to-frame motion
- Mouth-region motion
- Eye-region motion
- Eyes-over-mouth motion ratio

Used to detect unnatural motion decoupling (e.g., speaking mouth with static eyes).

### 7.2 Face Stability Evidence
- Face detection missing ratio
- Bounding box center jitter (normalized)
- Bounding box size jitter (normalized)

High instability may indicate compositing or tracking failure.

### 7.3 Blink-Related Evidence
- Eye openness proxy over time
- Blink-like event count (strict heuristic)
- Blink dip fraction
- Eye openness range

A specific experiment focused on detecting **absence of blinking**, a known deepfake artifact.

### 7.4 Boundary / Compositing Evidence
- Edge strength at face boundary
- Edge strength in background region
- Log-ratio calibration (face vs background)

Designed to detect mask-edge artifacts without requiring segmentation.

---

## 8. Blink Detection Experiment

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

## 9. Prompt Engineering

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

## 10. LLM Integration

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

---

## 11. Experimental Findings

### Key Observations
1. Stable heuristics often bias toward REAL.
2. Blink absence is surprisingly hard to detect reliably.
3. Boundary evidence is easily misinterpreted.
4. LLMs reflect the quality of evidence provided.

### Important Result
> Even with explicit evidence, an LLM may reasonably conclude UNCERTAIN.

This aligns with the professor’s original skepticism.

---

## 12. Limitations

- No facial landmarks or EAR metrics
- No optical flow
- No audio-visual alignment
- Limited frame sampling
- Heuristics are brittle by nature

These limitations are explicitly documented rather than hidden.

---

## 13. Conclusion

This project demonstrates:
- a principled attempt at LLM-assisted deepfake detection,
- a transparent, inspectable pipeline,
- honest analysis of failure modes.

The inability to reliably detect manipulation is itself a valuable conclusion.

---

## 14. How to Run

### Environment Setup
```bash
export AZURE_OPENAI_ENDPOINT=...
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_DEPLOYMENT_NAME=...
export AZURE_OPENAI_API_VERSION=...
```

### Run Example
```bash
python -m src.deepfake_detector.cli detect   --video assets/videos/example.mp4   --out runs/example   --llm azure   --max-keyframes 2
```

---

## 15. Final Note

This work prioritizes **scientific honesty over success metrics**.

From an academic standpoint, the project succeeds by:
- exploring a difficult question,
- iterating thoughtfully,
- documenting limitations clearly.

# Test comment
# Another test
