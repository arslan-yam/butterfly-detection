import json
import tempfile
import unittest
from pathlib import Path

from models.data import prepare
from models.io import file_sha, read_jsonl, write_json, write_jsonl
from models.rescore import CONFIGS, rescore


class RescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "reviewed.jsonl"
        self.original = self.root / "raw"
        self.output = self.root / "golden"
        self.box = dict(left=0.1, top=0.2, width=0.3, height=0.4)
        write_jsonl(self.source, [dict(photo_id=1, image="a.jpg", status="butterflies", boxes=[self.box]),
                                 dict(photo_id=2, image="b.jpg", status="no_butterflies", boxes=[])])
        prepare([self.source], self.original, "raw", 2)
        model_ids = ["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-tiny",
                     "google/owlv2-base-patch16-ensemble", "google/owlv2-base-patch16-ensemble",
                     "Qwen/Qwen3-VL-2B-Instruct"]
        self.hashes = {}
        for config, model_id in zip(CONFIGS, model_ids):
            folder = self.original / config
            write_jsonl(folder / "predictions.jsonl", [dict(photo_id=1, boxes=[self.box], error=None),
                                                       dict(photo_id=2, boxes=[], error=None)])
            self.hashes[config] = file_sha(folder / "predictions.jsonl")
            write_json(folder / "run.json", dict(config=dict(name=config, model_id=model_id),
                       resolved_revision="a" * 40, manifest_sha256=file_sha(self.original / "manifest.jsonl"),
                       predictions_sha256=self.hashes[config], limit=None))

    def tearDown(self):
        self.temp.cleanup()

    def test_updated_labels_and_reordered_export_reuse_original_predictions(self):
        rows = read_jsonl(self.source)[::-1]
        rows[1].update(status="no_butterflies", boxes=[])
        write_jsonl(self.source, rows)
        rescore([self.source], self.original, self.output)
        self.assertEqual(file_sha(self.output / "manifest.jsonl"), file_sha(self.original / "manifest.jsonl"))
        for config in CONFIGS:
            self.assertEqual(file_sha(self.original / config / "predictions.jsonl"), self.hashes[config])
            report = json.loads((self.output / config / "metrics.json").read_text())
            self.assertEqual(report["annotation_stage"], "golden")
            self.assertEqual(report["iou_0_5"]["fp"], 1)
            self.assertEqual(report["iou_0_5"]["tp"], 0)
        selection = json.loads((self.output / "selection.json").read_text())
        self.assertEqual(selection["status"], "ready")
        self.assertEqual(read_jsonl(self.original / "annotations.jsonl")[0]["boxes"], [self.box])

    def test_changed_image_is_rejected(self):
        rows = read_jsonl(self.source)
        rows[0]["image"] = "different.jpg"
        write_jsonl(self.source, rows)
        with self.assertRaises(ValueError):
            rescore([self.source], self.original, self.output)
        self.assertFalse(self.output.exists())

    def test_missing_photo_is_rejected(self):
        write_jsonl(self.source, read_jsonl(self.source)[:1])
        with self.assertRaises(ValueError):
            rescore([self.source], self.original, self.output)

    def test_error_cannot_be_hidden_by_new_labels(self):
        folder = self.original / "qwen"
        rows = read_jsonl(folder / "predictions.jsonl")
        rows[0].update(boxes=None, error="generation failed")
        write_jsonl(folder / "predictions.jsonl", rows)
        with self.assertRaises(ValueError):
            rescore([self.source], self.original, self.output)

    def test_cannot_overwrite_original_run(self):
        with self.assertRaises(ValueError):
            rescore([self.source], self.original, self.original)
