from pathlib import Path

from src.deepfake_detector.llm_input.input_builder import build_llm_input
from src.deepfake_detector.prompts.prompt_builder import build_prompt_text
from src.deepfake_detector.llm.mock_client import MockLLMClient
from src.deepfake_detector.decision.parser import parse_llm_output


ROOT = Path(__file__).resolve().parents[1]
manifest = ROOT / "assets" / "frames" / "man_speak_blink" / "frames_manifest.json"
evidence = ROOT / "assets" / "frames" / "man_speak_blink" / "evidence_basic.json"

llm_input = build_llm_input(
    manifest_path=str(manifest),
    evidence_path=str(evidence),
    max_keyframes=8,
)

prompt = build_prompt_text(llm_input)
print(prompt)

client = MockLLMClient()
resp = client.generate(prompt=prompt, image_paths=[kf["path"] for kf in llm_input["keyframes"]])
decision = parse_llm_output(resp.raw_text)
print(decision)
print(resp.raw_text)
