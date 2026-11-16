from __future__ import annotations

import argparse
import json

from .data import download, prepare
from .evaluation import evaluate, export_contest, leaderboard


def main():
    parser = argparse.ArgumentParser(description="Butterfly detection: inference, evaluation, contest export")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Separate model inputs from annotations")
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stage", choices=["raw", "golden", "unlabeled"], required=True)
    p.add_argument("--expected-count", type=int)
    p = commands.add_parser("download", help="Download/verify images without loading models")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--retries", type=int, default=3)
    p = commands.add_parser("lock", help="Resolve model revisions once before comparison")
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("run", help="Run one model; checkpoints after every image")
    p.add_argument("--manifest", required=True)
    p.add_argument("--images", required=True, help="images.jsonl from download")
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, help="Smoke test only; cannot be selected for final comparison")
    p = commands.add_parser("evaluate", help="Full-dataset object detection metrics")
    p.add_argument("--annotations", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--allow-errors", action="store_true", help="Diagnostic lower bound only")
    p = commands.add_parser("leaderboard")
    p.add_argument("--metrics", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("export")
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--expected-count", type=int, default=400)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "prepare":
            prepare(**args)
        elif command == "download":
            if args["retries"] < 1:
                raise ValueError("retries must be positive")
            result = download(**args)
            return int(any(row["error"] for row in result))
        elif command == "run":
            from .runner import run
            args["config_path"] = args.pop("config")
            result = run(**args)
            return int(any(row["error"] for row in result))
        elif command == "lock":
            from .runner import lock_configs
            lock_configs(**args)
        elif command == "evaluate":
            result = evaluate(**args)
            print(json.dumps({k: v for k, v in result.items() if k != "groups"}, ensure_ascii=False, indent=2))
        elif command == "leaderboard":
            leaderboard(**args)
        elif command == "export":
            export_contest(**args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
