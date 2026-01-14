# Prompt Log: Project Development and Engineering

This document tracks the key prompts and design iterations used during the development of the Deepfake Detection project. It serves as a transparent record of how the system's architecture and forensic reasoning evolved through interactions with an AI agent.

---

## Phase 1: Architecture and Pipeline Design

### Prompt 1: Initial Vision
> "I want to build a deepfake detection system that doesn't just give a score, but provides interpretable evidence. The system should extract forensic signals (like motion, blinking, and boundary artifacts) from a video and then use an LLM to reason over those signals to make a final decision. Can you propose a modular architecture for this?"

**Outcome:**
- Defined the modular pipeline: `Video -> Frames -> Evidence -> Prompt -> LLM -> Decision`.
- Established the "artifact-first" philosophy where every stage saves its output to disk.

### Prompt 2: Evidence Contract
> "How should we structure the communication between the evidence extraction and the LLM? I want it to be deterministic and human-readable."

**Outcome:**
- Designed the `evidence_basic.json` and `frames_manifest.json` formats.
- Decided on using JSON as the intermediate contract.

---

## Phase 2: Forensic Signal Engineering

### Prompt 3: Blink Detection Heuristics
> "One common deepfake artifact is the absence of blinking. I need a way to detect this without using a heavy deep learning model. Can we use Laplacian variance on the eye region as a proxy for eye openness?"

**Outcome:**
- Implemented `src/deepfake_detector/evidence/blink.py`.
- Developed the "dip detection" heuristic to count blink-like events.

### Prompt 4: Handling Noise in Blinks
> "The blink detection is too noisy. Small movements are being counted as blinks. How can we make it more robust?"

**Outcome:**
- Added log-scaling to the openness metric.
- Implemented ROI (Region of Interest) calibration to handle slight head movements.
- Introduced strict "event" criteria (minimum dip depth and duration).

---

## Phase 3: Prompt Engineering and LLM Reasoning

### Prompt 5: First Reasoning Template
> "Create a prompt template that takes the extracted metrics (motion ratios, blink counts, face stability) and asks GPT-4 Vision to determine if a video is REAL or MANIPULATED. It must provide a 'Reason' field."

**Outcome:**
- Created the initial version of `src/deepfake_detector/prompts/prompt_builder.py`.
- Discovered that the LLM often ignored subtle numeric cues without explicit heuristic guidance in the prompt.

### Prompt 6: Adding Heuristic Guidance
> "The LLM is struggling to interpret the 'eyes-over-mouth motion ratio'. Let's add explicit 'interpretation guides' to the prompt, telling the LLM what typical ranges for REAL and MANIPULATED videos look like based on our experiments."

**Outcome:**
- Updated prompt builder to include: "Guidance: A ratio < 0.2 often indicates decoupled mouth motion (suspicious)."
- Significant improvement in reasoning consistency.

---

## Phase 4: Reliability and Packaging

### Prompt 7: Testing Strategy
> "We need to ensure this pipeline is stable. We have many components. How should we test this without calling the expensive Azure API every time?"

**Outcome:**
- Developed the `MockLLMClient` that mimics GPT-4's output format based on input evidence.
- Wrote 153 unit tests covering every transformation in the pipeline.

### Prompt 8: Final Documentation
> "We need a comprehensive README and User Guide. It should emphasize that this is a RESEARCH project and that failure to detect a deepfake is an honest result."

**Outcome:**
- Finalized `README.md` and `docs/USER_GUIDE.md` with sections on Research Motivation and Limitations.

---

## Phase 5: Refining Forensic Signals (Boundary & Stability)

### Prompt 9: Edge Artifact Detection
> "How can we detect compositing artifacts around the face without using a segmentation model? Can we compare the edge strength in a 'ring' around the face bounding box to the edge strength inside it?"

**Outcome:**
- Implemented the "boundary edge ratio" in `src/deepfake_detector/evidence/basic_signals.py`.
- Developed a calibration step that compares the face boundary ratio to a background boundary ratio to reduce false positives from camera noise or high-frequency backgrounds.

### Prompt 10: Face Stability as a Signal
> "Deepfakes often show jitter in the face placement. Can we track the bounding box coordinates and size across frames to detect this?"

**Outcome:**
- Added `face_bbox_center_jitter` and `face_bbox_size_jitter` metrics.
- Normalized these by the bounding box size to ensure the metric is scale-invariant.

---

## Phase 6: Multi-Agent and Multi-Step Reasoning Evolution

### Prompt 11: From Classification to Reasoning
> "The model is jumping to conclusions too fast. Let's change the prompt to require a 'Reasoning' step before the 'Label'. We want it to look at the motion, then the blinks, then the boundary, and only then decide."

**Outcome:**
- Updated `src/deepfake_detector/prompts/prompt_builder.py` to structure the instructions as a step-by-step analysis.
- Note: The current implementation uses a "Reason" field which combines these, but the internal instruction encourages sequential evaluation.

### Prompt 12: Handling Uncertainty
> "If the evidence is contradictory (e.g., natural blinking but high boundary jitter), how should the model respond? We should allow an UNCERTAIN label."

**Outcome:**
- Explicitly added `UNCERTAIN` to the allowed labels in the prompt and the `Decision` parser.
- Added heuristic guidance in the prompt on when to use UNCERTAIN (e.g., "when background has stronger edges than the face region").

---

## Summary of Evolution
- **Iteration 1:** Raw motion detection -> LLM. (Too vague)
- **Iteration 2:** Numeric metrics + Frames -> LLM. (Better, but inconsistent reasoning)
- **Iteration 3:** Numeric metrics + Heuristic Guidance + Keyframes -> LLM. (Current state: stable and interpretable)
- **Iteration 4:** Calibrated signals (Boundary, Stability) + Structured Reasoning. (Finalized research pipeline)

## Lessons Learned from Agent Interaction
1. **Numeric Range Matters:** LLMs don't inherently know what a "0.15 motion ratio" means. Providing explicit ranges (e.g., "< 0.2 is suspicious") is mandatory.
2. **Context is King:** The LLM needs to know the FPS and duration to judge if "0 blinks" is actually suspicious or just a very short video.
3. **Transparency over Accuracy:** In a research context, a well-reasoned "UNCERTAIN" is more valuable than a lucky "MANIPULATED" guess without evidence.
