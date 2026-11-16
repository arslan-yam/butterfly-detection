from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models.data import download, prepare
from models.evaluation import evaluate, export_contest, leaderboard
from models.geometry import iou, match_boxes, nms, normalize_xyxy, parse_qwen
from models.io import boxes_valid, file_sha, indexed, read_jsonl, write_json, write_jsonl
from models.runner import lock_configs, resolve_revision, run, validate_config


def box(left=0, width=0.5):
    return dict(left=left, top=0, width=width, height=0.5)


class GeometryTests(unittest.TestCase):
    def test_maximum_matching_requires_reassignment(self):
        # First GT matches both predictions; second GT can only match first prediction.
        a, b = box(0), box(0.2)
        self.assertEqual(match_boxes([a, b], [box(0.1), box(0)], 0.5), dict(tp=2, fp=0, fn=0))

    def test_duplicates_are_false_positives(self):
        self.assertEqual(match_boxes([box()], [box(), box()], 0.5), dict(tp=1, fp=1, fn=0))
        self.assertEqual(len(nms([dict(box(), score=0.2), dict(box(), score=0.8)], 0.5)), 1)
        self.assertEqual(nms([dict(box(), score=0.2), dict(box(), score=0.8)], 0.5)[0]["score"], 0.8)

    def test_iou_threshold_and_empty(self):
        self.assertEqual(iou(box(), box()), 1)
        self.assertEqual(match_boxes([], [box()], 0.75), dict(tp=0, fp=1, fn=0))
        self.assertEqual(match_boxes([box()], [box(0.1)], 0.75)["tp"], 0)
        self.assertEqual(match_boxes([box()], [box(0.1)], 0.5)["tp"], 1)

    def test_qwen_valid_and_empty(self):
        self.assertEqual(parse_qwen("[]"), [])
        parsed = parse_qwen('```json\n[{"bbox_format":"xyxy","box":[100,200,800,900]}]\n```')
        self.assertAlmostEqual(parsed[0]["width"], 0.7)
        self.assertEqual(parsed[0]["top"], 0.2)

    def test_qwen_invalid_is_error(self):
        for text in ('No butterflies', '{"boxes":[]}', '[{"bbox_format":"xyxy","box":[20,0,10,30]}]',
                     '[{"bbox_format":"xyxy","box":[0,0,1001,30]}]',
                     '[{"bbox_format":"xyxy","box":[false,0,10,30]}]', '[] extra', '[,]', '[ , ]'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_qwen(text)

    def test_qwen_trailing_comma_seen_in_real_inference(self):
        raw = '```json\n[\n\t{"bbox_format": "xyxy", "box": [98, 270, 867, 881]},\t\n]\n```'
        result = parse_qwen(raw)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["left"], 0.098)
        # Punctuation inside a string must not be altered to pass validation.
        with self.assertRaises(ValueError):
            parse_qwen('[{"bbox_format":"xyxy,}", "box":[0,0,10,10]}]')

    def test_qwen_supplementary_label_does_not_change_coordinates(self):
        raw = '[{"bbox_format":"xyxy","box":[388,388,695,565],"label":"adult butterfly"},'
        raw += '{"bbox_format":"xyxy","box":[388,388,695,565],"label":"adult moth"}]'
        result = parse_qwen(raw)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], result[1])
        self.assertAlmostEqual(result[0]["left"], 0.388)
        self.assertEqual(len(nms(result, 0.5)), 1)

    def test_qwen_coordinates_duplicated_into_format_tag(self):
        result = parse_qwen('[{"bbox_format":[78,70,303,282],"box":[78,70,303,282],"label":"adult moth"}]')
        self.assertEqual(result[0]["left"], 0.078)
        self.assertEqual(result[0]["top"], 0.07)
        for raw in ('[{"bbox_format":"xywh","box":[78,70,303,282]}]',
                    '[{"bbox_format":[1,2,3,4],"box":[78,70,303,282]}]'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_qwen(raw)

    def test_invalid_normalized_boxes(self):
        for value in (None, [box(width=0)], [box(left=0.8)], [box(width=float("nan"))], [box(width=True)]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                boxes_valid(value)

    def test_detector_padding_boxes_and_clipping(self):
        self.assertIsNone(normalize_xyxy([120, 0, 150, 10], 100, 100))
        self.assertEqual(normalize_xyxy([-10, 0, 50, 50], 100, 100), box())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.sources = self.root / "source.jsonl"
        write_jsonl(self.sources, [dict(photo_id="1", image="a.jpg", status="butterflies", boxes=[box()]),
                                  dict(photo_id=2, image="b.jpg", status="no_butterflies", boxes=[])])
        self.dataset = self.root / "dataset"
        prepare([self.sources], self.dataset, "golden", 2)
        self.manifest = self.dataset / "manifest.jsonl"
        self.annotations = self.dataset / "annotations.jsonl"
        self.predictions = self.root / "predictions.jsonl"
        write_jsonl(self.predictions, [dict(photo_id=1, boxes=[box()], error=None, elapsed_seconds=0.5),
                                      dict(photo_id="2", boxes=[box()], error=None, elapsed_seconds=1.0)])

    def tearDown(self):
        self.temp.cleanup()

    def test_prepare_separates_annotations(self):
        manifest = read_jsonl(self.manifest)
        self.assertNotIn("boxes", manifest[0])
        self.assertNotIn("status", manifest[0])
        self.assertEqual(len(manifest), 2)

    def test_duplicate_numeric_ids_rejected(self):
        with self.assertRaises(ValueError):
            indexed([dict(photo_id="01"), dict(photo_id=1)])
        with self.assertRaises(ValueError):
            prepare([self.sources, self.sources], self.root / "bad", "raw")

    def test_unresolved_annotations_rejected(self):
        rows = read_jsonl(self.sources)
        rows[0]["status"] = "image_error"
        write_jsonl(self.sources, rows)
        with self.assertRaises(ValueError):
            prepare([self.sources], self.root / "bad", "golden")

    def test_expected_count_rejected(self):
        with self.assertRaises(ValueError):
            prepare([self.sources], self.root / "bad", "raw", 400)

    def test_metrics_micro_and_negative_fp(self):
        result = evaluate(self.annotations, self.predictions, self.manifest, self.root / "metrics")
        self.assertEqual(result["iou_0_5"]["tp"], 1)
        self.assertAlmostEqual(result["iou_0_5"]["f1"], 2 / 3)
        self.assertEqual(result["negative_fp_boxes"], 1)
        self.assertEqual(result["negative_photos_with_fp"], 1)
        self.assertEqual(result["mean_seconds"], 0.75)
        self.assertFalse(result["eligible_for_selection"])  # No run provenance.

    def test_errors_fail_default_and_remain_in_diagnostics(self):
        write_jsonl(self.predictions, [dict(photo_id=1, boxes=None, error="OOM"), dict(photo_id=2, boxes=[], error=None)])
        with self.assertRaises(ValueError):
            evaluate(self.annotations, self.predictions, self.manifest, self.root / "metrics")
        result = evaluate(self.annotations, self.predictions, self.manifest, self.root / "metrics", True)
        self.assertEqual(result["iou_0_5"]["fn"], 1)
        self.assertEqual(result["photos"], 2)
        self.assertFalse(result["eligible_for_selection"])
        with self.assertRaises(ValueError):
            export_contest(self.manifest, self.predictions, self.root / "submission.jsonl", 2)

    def test_missing_ids_rejected(self):
        write_jsonl(self.predictions, read_jsonl(self.predictions)[:1])
        with self.assertRaises(ValueError):
            evaluate(self.annotations, self.predictions, self.manifest, self.root / "metrics")
        with self.assertRaises(ValueError):
            export_contest(self.manifest, self.predictions, self.root / "submission.jsonl", 2)

    def test_changed_annotations_rejected(self):
        write_jsonl(self.annotations, [dict(photo_id=1, boxes=[]), dict(photo_id=2, boxes=[])])
        with self.assertRaises(ValueError):
            evaluate(self.annotations, self.predictions, self.manifest, self.root / "metrics")

    def test_export_strips_extra_fields(self):
        output = self.root / "submission.jsonl"
        export_contest(self.manifest, self.predictions, output, 2)
        rows = read_jsonl(output)
        self.assertEqual(set(rows[0]), {"photo_id", "boxes", "error"})
        self.assertEqual(rows[0]["photo_id"], 1)
        with self.assertRaises(ValueError):
            export_contest(self.manifest, self.predictions, output)

    def test_raw_leaderboard_does_not_select_winner(self):
        prepare([self.sources], self.dataset, "raw", 2)
        evaluate(self.annotations, self.predictions, self.manifest, self.root / "metrics")
        leaderboard([self.root / "metrics/metrics.json"], self.root / "board")
        selection = json.loads((self.root / "board/selection.json").read_text())
        self.assertIsNone(selection["best"])
        self.assertEqual(selection["status"], "provisional")

    def test_download_local_and_resume_checks_hash(self):
        from PIL import Image
        for name in ("a.jpg", "b.jpg"):
            Image.new("RGB", (12, 8), "green").save(self.dataset / name)
        records = download(self.manifest, self.root / "cache", retries=1)
        self.assertTrue(all(r["error"] is None for r in records))
        path = self.root / "cache" / records[0]["path"]
        path.write_bytes(b"corrupted")
        records2 = download(self.manifest, self.root / "cache", retries=1)
        self.assertEqual(records[0]["sha256"], records2[0]["sha256"])
        self.assertEqual(file_sha(path), records[0]["sha256"])

    def test_download_failure_is_explicit(self):
        records = download(self.manifest, self.root / "cache", retries=1)
        self.assertTrue(all(r["error"] for r in records))

    def test_configs_and_one_factor_experiment(self):
        configs = Path(__file__).parents[1] / "configs"
        for path in configs.glob("*.json"):
            validate_config(json.loads(path.read_text()))
        baseline = json.loads((configs / "dino.json").read_text())
        variant = json.loads((configs / "dino_threshold035.json").read_text())
        changed = {k for k in baseline if baseline[k] != variant[k]}
        self.assertEqual(changed - {"name", "hypothesis"}, {"threshold"})
        self.assertEqual(baseline["prompt"], "butterfly. moth.")
        self.assertEqual(baseline["threshold"], 0.25)
        self.assertEqual(variant["threshold"], 0.35)
        owl = json.loads((configs / "owlv2.json").read_text())
        owl_variant = json.loads((configs / "owlv2_threshold020.json").read_text())
        changed = {k for k in owl if owl[k] != owl_variant[k]}
        self.assertEqual(changed - {"name", "hypothesis"}, {"threshold"})
        self.assertEqual(owl_variant["threshold"] - owl["threshold"], 0.1)

    def test_runner_resume_retries_only_failures_and_locks_inputs(self):
        from PIL import Image
        for name in ("a.jpg", "b.jpg"):
            Image.new("RGB", (12, 8)).save(self.dataset / name)
        download(self.manifest, self.root / "cache", retries=1)
        config_path = Path(__file__).parents[1] / "configs/dino.json"
        config = json.loads(config_path.read_text())
        config_path = self.root / "config.json"
        write_json(config_path, config)
        backend = SimpleNamespace(synchronize=lambda: None)
        backend.predict = unittest.mock.Mock(side_effect=[([box()], None), ValueError("temporary failure")])
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), manual_seed=lambda s: None,
                                     version=SimpleNamespace(cuda=None))
        fake_hub = SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="abc123")))
        arguments = dict(manifest=self.manifest, images=self.root / "cache/images.jsonl", config_path=config_path,
                         output=self.root / "run")
        with patch.dict("sys.modules", {"torch": fake_torch, "huggingface_hub": fake_hub}), \
                patch("models.runner.HuggingFaceBackend", return_value=backend):
            first = run(**arguments)
            self.assertEqual(sum(bool(r["error"]) for r in first), 1)
            backend.predict.reset_mock(side_effect=True)
            backend.predict.return_value = ([], None)
            second = run(**arguments, resume=True)
            self.assertEqual(backend.predict.call_count, 1)
            self.assertTrue(all(r["error"] is None for r in second))
            with self.assertRaises(ValueError):
                run(**arguments)
            config["threshold"] = 0.4
            write_json(config_path, config)
            with self.assertRaises(ValueError):
                run(**arguments, resume=True)
        self.assertTrue((self.root / "run/errors.jsonl").exists())

    def test_lock_resolves_shared_revision_once(self):
        config_dir = Path(__file__).parents[1] / "configs"
        sources = []
        for name in ("dino.json", "dino_threshold035.json"):
            config = json.loads((config_dir / name).read_text())
            config["revision"] = "main"
            source = self.root / name
            write_json(source, config)
            sources.append(source)
        api = SimpleNamespace(model_info=unittest.mock.Mock(return_value=SimpleNamespace(sha="abc123")))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(HfApi=lambda: api)}):
            lock_configs(sources, self.root / "locked")
            self.assertEqual(api.model_info.call_count, 1)
            lock_configs(sources, self.root / "locked")
            self.assertEqual(api.model_info.call_count, 1)

    def test_pinned_revision_needs_no_api(self):
        api = unittest.mock.Mock(side_effect=AssertionError("Must not contact API for fixed SHA"))
        with patch.dict("sys.modules", {"huggingface_hub": SimpleNamespace(HfApi=api)}):
            self.assertEqual(resolve_revision("model", "a" * 40), "a" * 40)


if __name__ == "__main__":
    unittest.main()
