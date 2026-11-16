"""Freeze pip's installation overlay for the exact Docker base (stdlib only)."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("report", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
report = json.loads(args.report.read_text())
lines = ["# Linux amd64 / Python 3.11; overlay for the digest-pinned Docker base.",
         "# Inherited dependencies are fixed by FROM; install with --no-deps --require-hashes.",
         "# Generated from pip --dry-run --report, not an independent virtualenv lock."]
for item in sorted(report["install"], key=lambda item: item["metadata"]["name"].lower()):
    name, version = item["metadata"]["name"], item["metadata"]["version"]
    sha = item["download_info"]["archive_info"]["hashes"]["sha256"]
    if not item["download_info"]["url"].endswith(".whl") or len(sha) != 64:
        raise ValueError(f"Expected a SHA256-verified wheel: {name}")
    lines.append(f"{name}=={version} --hash=sha256:{sha}")
args.output.write_text("\n".join(lines) + "\n")
