# Synthetic Video Generation Design

## Purpose
This document defines the synthetic video datasets used to evaluate the LLM-based deepfake detector.
It focuses on *experiment design*: each video variant introduces a controlled artifact that can be
described in natural language and used in prompt engineering and evaluation.

This document is not an API reference. Developer usage (how to run the generator and parameters)
is documented in the generator module docstrings and (later) the CLI/User Guide.

## Common Parameters (All Variants)
All variants share the following configurable parameters:
- width, height: output resolution (default 512x512)
- fps: frames per second (default 30)
- duration_seconds: video duration (default 10 seconds)
- background: solid color, constant lighting
- seed: accepted for determinism and future stochastic variants

All videos are deterministic given the same parameters.

## Output Artifacts (All Variants)
Each generation run outputs:
1) An MP4 video file (e.g., fake_v0.mp4)
2) A ground-truth description TXT file (e.g., fake_v0_description.txt)

The TXT file is for documentation/evaluation only and must not be provided to the LLM as input.

## Variants Overview
Variants are defined as controlled experiments. Each variant introduces *one* dominant artifact.

- v0 (FAKE): Mouth moves continuously; eyes are perfectly static (no blinking).
  Primary artifact: lack of blinking despite mouth motion.

Planned variants (v1+), to be implemented after v0 baseline works:
- v1 (FAKE): Blinking exists but is perfectly periodic (unnatural regularity).
- v2 (FAKE): Mouth motion is jittery/discontinuous frame-to-frame.
- v3 (FAKE): Lighting flicker inconsistent with a static scene.
- v4 (REAL-ish baseline): Mouth motion + naturalistic random blinking (no intentional artifact).

Each variant must include:
- a short natural-language ground truth description of the artifact
- a clear statement of what behavior is being tested

---

## Variant v0 Specification 

### Concept

----------
A single face-like object is drawn using simple geometric shapes:
- Head: circle
- Eyes: two small circles (perfectly static across all frames)
- Mouth: ellipse/arc that opens and closes smoothly over time

The intentional fake artifact in v0:
- The mouth moves continuously across the video while the eyes never blink and
  remain perfectly static for the entire duration.

No additional artifacts are introduced in v0 (no head movement, no lighting
changes, no random noise, no eye movement).

Outputs
-------
This module must generate two artifacts:
1) Video file (MP4): fake_v0.mp4
2) Ground-truth description (TXT): fake_v0_description.txt

The ground-truth description is for documentation and evaluation only; it is
NOT intended to be sent to the LLM during inference.

Determinism
-----------
The generator must be deterministic:
- Given the same parameters and seed, it must produce identical frames.
- v0 should avoid randomness entirely; a seed is still accepted to support
  future versions (v1+), but v0 behavior must remain deterministic even without
  random components.

v0 Video Specification (Default Parameters)
-------------------------------------------
- Resolution: 512 x 512
- FPS: 30
- Duration: 10 seconds
- Total frames: 300
- Camera: static
- Background: solid color
- Lighting: constant

Scene: one face-like object centered on the canvas.

Motion Rules
------------
What DOES move:
- Mouth opens/closes smoothly over time using a sinusoidal function.

What DOES NOT move (primary fake signal):
- Eyes never blink: eye appearance/size/position remain identical in all frames.

Strict Constraints (v0)
-----------------------
- Do NOT add head movement or camera movement
- Do NOT add lighting changes
- Do NOT add texture noise or GAN-like artifacts
- Do NOT add eye motion or blinking
- Do NOT add multiple faces
- Do NOT randomize behavior (keep deterministic)

API Contract (Planned)
----------------------
Expected public function:

    def generate_fake_v0(
        out_video_path: str,
        out_description_path: str,
        *,
        width: int = 512,
        height: int = 512,
        fps: int = 30,
        duration_seconds: int = 10,
        background_rgb: tuple[int, int, int] = (40, 40, 40),
        seed: int = 123,
    ) -> None:
        \"\"\"Generate the v0 synthetic deepfake video and a ground-truth
        description file.

        Parameters
        ----------
        out_video_path:
            Destination path for the output MP4 video.
        out_description_path:
            Destination path for the ground-truth description text file.
        width, height:
            Output frame dimensions in pixels.
        fps:
            Frames per second.
        duration_seconds:
            Video length in seconds. Total frames = fps * duration_seconds.
        background_rgb:
            Solid background color.
        seed:
            RNG seed for future versions. v0 should not rely on randomness.

        Behavior
        --------
        - Draw a simple face using geometric primitives.
        - Apply sinusoidal mouth opening across frames.
        - Keep eyes perfectly static (no blinking).

        Outputs
        -------
        - Writes an MP4 video to out_video_path.
        - Writes a TXT description to out_description_path.

        Error Handling
        --------------
        - Raise ValueError for invalid parameters (non-positive fps, dimensions,
          duration, etc.).
        - Raise IOError/OSError on write failures.

        \"\"\"

Implementation Notes (Planned)
------------------------------
- Use OpenCV (cv2.VideoWriter) to write MP4.
- Use numpy arrays for frame buffers.
- Represent mouth opening as a continuous value in [0, 1] per frame:
      open_factor = (sin(2*pi*t/period) + 1)/2
  where t is frame index / fps.
- Map open_factor to mouth height / curvature while keeping position fixed.
- Keep all other features constant across frames.

Ground Truth Description (v0)
-----------------------------
The generator must write the following meaning (exact wording can be stable):

"This video was synthetically generated.
The mouth shows continuous, smooth motion over time.
The eyes remain completely static and do not blink at any point.
This inconsistency is intentional and represents an unnatural facial behavior."

Verification (Planned)
----------------------
A simple manual check should be possible:
- Scrub through frames and confirm:
  - mouth changes shape smoothly
  - eyes are identical in every frame
  - background and head remain fixed

"""
