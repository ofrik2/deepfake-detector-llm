import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.deepfake_detector.cli import main
from src.deepfake_detector.detectors.base import BaseDetector, DetectorResult
from src.deepfake_detector.detectors.registry import (
    _REGISTRY,
    discover_plugins,
    get_detector,
    list_detectors,
    register_detector,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear the registry before each test."""
    original_registry = _REGISTRY.copy()
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original_registry)


def test_manual_registration():
    @register_detector("dummy")
    class DummyDetector(BaseDetector):
        @property
        def name(self):
            return "dummy"

        def detect(self, video_path, out_dir, config=None):
            return DetectorResult(label="REAL", rationale="test")

    assert "dummy" in list_detectors()
    assert get_detector("dummy") == DummyDetector


def test_discovery_external_plugin():
    with tempfile.TemporaryDirectory() as tmpdir:
        plugin_path = Path(tmpdir) / "my_plugin.py"
        plugin_path.write_text(
            """
from src.deepfake_detector.detectors.base import BaseDetector, DetectorResult
from src.deepfake_detector.detectors.registry import register_detector

@register_detector("external")
class ExternalDetector(BaseDetector):
    @property
    def name(self): return "external"
    def detect(self, video_path, out_dir, config=None):
        return DetectorResult(label="MANIPULATED", rationale="external plugin")
"""
        )

        # Test loading via root plugins/ folder
        with patch("pathlib.Path.cwd", return_value=Path(tmpdir)):
            # Need to fake that "plugins" folder exists in cwd
            plugins_dir = Path(tmpdir) / "plugins"
            plugins_dir.mkdir()
            shutil.move(str(plugin_path), str(plugins_dir / "my_plugin.py"))

            discover_plugins()
            assert "external" in list_detectors()

            detector_cls = get_detector("external")
            detector = detector_cls()
            result = detector.detect("vid.mp4", "out/")
            assert result.label == "MANIPULATED"


def test_cli_list_detectors(capsys):
    @register_detector("cli_test")
    class CliTestDetector(BaseDetector):
        @property
        def name(self):
            return "cli_test"

        def detect(self, video_path, out_dir, config=None):
            return None

    with patch("sys.argv", ["prog", "--list-detectors"]):
        main()

    captured = capsys.readouterr()
    assert "- cli_test" in captured.out


def test_cli_detect_with_custom_detector(capsys):
    @register_detector("custom")
    class CustomDetector(BaseDetector):
        @property
        def name(self):
            return "custom"

        def detect(self, video_path, out_dir, config=None):
            return DetectorResult(
                label="UNCERTAIN", rationale="custom detector rationale", metadata={"foo": "bar"}
            )

    with patch(
        "sys.argv", ["prog", "detect", "--video", "v.mp4", "--out", "o/", "--detector", "custom"]
    ):
        main()

    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["label"] == "UNCERTAIN"
    assert output["rationale"] == "custom detector rationale"
    assert output["metadata"]["foo"] == "bar"
