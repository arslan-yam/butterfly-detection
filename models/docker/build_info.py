"""Run inside the image at build time; records exact copied implementation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, "/app")
from models.io import digest, file_sha

root = Path("/app")
sources = {str(p.relative_to(root)): file_sha(p) for p in sorted((root / "models").rglob("*"))
           if p.is_file() and "__pycache__" not in p.parts}
info = dict(source_sha256=digest(sources), source_files=sources,
            requirements_sha256=file_sha(root / "requirements.lock"),
            environment_sha256=file_sha(root / "environment.freeze.txt"),
            base="pytorch/pytorch@sha256:68c022c2f4627943a6f3e574cfd2c8ae4256210d5f66ae2b117942e0a8d4fa9d")
(root / "build-info.json").write_text(json.dumps(info, indent=2) + "\n")
