import os
import importlib.util
import sys
import logging
from pathlib import Path
from typing import Dict, Type, List, Optional
from .base import BaseDetector

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, Type[BaseDetector]] = {}

def register_detector(name: str):
    """Decorator to register a detector class."""
    def decorator(cls: Type[BaseDetector]):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_detector(name: str) -> Optional[Type[BaseDetector]]:
    """Return the detector class for a given name."""
    return _REGISTRY.get(name)

def list_detectors() -> List[str]:
    """Return a list of registered detector names."""
    return sorted(list(_REGISTRY.keys()))

def discover_plugins():
    """
    Discover and load built-in and external plugins.
    """
    # 1. Load built-in detectors
    # For built-ins, we can just import them if they are in the package
    try:
        from . import llm_detector
    except ImportError:
        # Fallback to dynamic loading if not in package context
        built_in_dir = Path(__file__).parent
        _load_from_dir(built_in_dir)

    # 2. Load from plugins/ folder at repo root
    root_plugins = Path.cwd() / "plugins"
    if root_plugins.exists() and root_plugins.is_dir():
        _load_from_dir(root_plugins)

    # 3. Load from env var DEEPFAKE_DETECTOR_PLUGINS_PATH
    env_path = os.getenv("DEEPFAKE_DETECTOR_PLUGINS_PATH")
    if env_path:
        for path_str in env_path.split(os.pathsep):
            path = Path(path_str)
            if path.exists() and path.is_dir():
                _load_from_dir(path)

def _load_from_dir(directory: Path):
    """Safely load python modules from a directory."""
    for file in directory.glob("*.py"):
        if file.name == "__init__.py" or file.name == "base.py" or file.name == "registry.py":
            continue
        
        try:
            # Generate a unique module name for the plugin
            module_name = f"detector_plugin_{file.stem}"
            spec = importlib.util.spec_from_file_location(module_name, file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
        except Exception as e:
            print(f"Warning: Failed to load plugin from {file}: {e}", file=sys.stderr)
