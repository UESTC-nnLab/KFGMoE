"""Metric definitions, COCO equivalence and explicit empty-case behavior."""
import contextlib
import io
import json
import unittest

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from kfgmoe.metrics import evaluate, summarize_envelope


def prediction(box, score):
    return {"bbox": box, "score": score, "category_id": 1}


def row(image_id=1, domain="DAUB-H", gt=None, predictions=None):
    return {"image_id": image_id, "domain": domain, "width": 64, "height": 64,
            "gt_boxes": [[0, 0, 10, 10]] if gt is None else gt,
            "predictions": [] if predictions is None else predictions}


class MetricsTests(unittest.TestCase):
    def test_paper_summary_is_not_fixed_threshold(self):
        result = evaluate([row(predictions=[prediction([40, 40, 10, 10], .8),
                                           prediction([0, 0, 10, 10], .1)])])["macro"]
        self.assertAlmostEqual(result["Pr"], .5)
        self.assertEqual(result["Rc"], 1.)
        self.assertAlmostEqual(result["F1"], 2 / 3)
        self.assertEqual(result["Fa"], 1.)

    def test_empty_prediction_and_empty_ground_truth(self):
        value = evaluate([row()])["per_domain"]["DAUB-H"]
        self.assertEqual(value["mAP50"], 0.)
        self.assertEqual(value["Rc"], 0.)
        self.assertIsNone(value["Pr"])
        self.assertIsNone(value["F1"])
        self.assertEqual(value["Fa"], 0.)
        value = evaluate([row(gt=[], predictions=[prediction([0, 0, 5, 5], .5)])])
        self.assertIsNone(value["macro"]["mAP50"])
        self.assertIsNone(value["macro"]["Rc"])
        self.assertEqual(value["macro"]["Fa"], 1.)
        json.dumps(value, allow_nan=False)

    def test_multiple_targets_fa_first_gt_iou_tie(self):
        # First detection overlaps both targets equally. Historical Fa chooses
        # the first GT; COCO chooses the last and can match both detections.
        value = evaluate([row(gt=[[0, 0, 10, 10], [4, 0, 14, 10]], predictions=[
            prediction([2, 0, 10, 10], .9), prediction([0, 0, 10, 10], .8)])])["macro"]
        self.assertAlmostEqual(value["mAP50"], 1.)
        self.assertEqual(value["Rc"], 1.)
        self.assertEqual(value["Fa"], 1.)

    def test_macro_is_mean_domain_f1(self):
        rows = [row(domain="DAUB-H", gt=[[0, 0, 10, 10], [20, 20, 30, 30]],
                    predictions=[prediction([0, 0, 10, 10], .9)]),
                row(domain="IRDST-R", predictions=[prediction([40, 40, 10, 10], .8),
                                                    prediction([0, 0, 10, 10], .1)])]
        value = evaluate(rows)["macro"]
        self.assertAlmostEqual(value["Pr"], .75)
        self.assertAlmostEqual(value["Rc"], .75)
        self.assertAlmostEqual(value["F1"], 2 / 3)
        rows.append(row(domain="empty", gt=[]))
        self.assertIsNone(evaluate(rows)["macro"]["F1"])

    def test_exclusive_prefix(self):
        envelope = np.linspace(1., 0., 101)
        value = summarize_envelope(envelope, .5)
        self.assertEqual(value["precision_prefix_length"], 50)
        self.assertAlmostEqual(value["Pr"], envelope[:50].mean())
        self.assertIsNone(summarize_envelope(envelope, .009)["Pr"])

    def test_batched_all_iou_matches_official_coco_with_score_ties(self):
        rows = [row(3, gt=[], predictions=[prediction([0, 0, 10, 10], .5)]),
                row(1, predictions=[prediction([1, 1, 10, 10], .5)]), row(2),
                row(4, predictions=[prediction([0, 0, 10, 10], .7)])]
        images, annotations, predictions = [], [], []
        for value in rows:
            images.append({"id": value["image_id"], "width": 64, "height": 64})
            for x1, y1, x2, y2 in value["gt_boxes"]:
                annotations.append({"id": len(annotations) + 1, "image_id": value["image_id"],
                                    "category_id": 1, "bbox": [x1, y1, x2-x1, y2-y1],
                                    "area": (x2-x1)*(y2-y1), "iscrowd": 0})
            predictions.extend({"image_id": value["image_id"], **p} for p in value["predictions"])
        with contextlib.redirect_stdout(io.StringIO()):
            truth = COCO()
            truth.dataset = {"info": {}, "images": images, "annotations": annotations,
                             "categories": [{"id": 1, "name": "target"}]}
            truth.createIndex()
            reference = COCOeval(truth, truth.loadRes(predictions), "bbox")
            reference.evaluate()
            reference.accumulate()
        precision = reference.eval["precision"][:, :, 0, 0, -1]
        for batch_size in (1, 2, 128):
            value = evaluate(rows, batch_size=batch_size)["per_domain"]["DAUB-H"]
            np.testing.assert_array_equal(value["PR101"], precision[0])
            self.assertEqual(value["mAP50"], float(precision[0].mean()))
            self.assertEqual(value["mAP75"], float(precision[5].mean()))
            self.assertEqual(value["mAP50_95"], float(precision.mean()))

    def test_duplicate_and_invalid_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluate([row(), row()])
        with self.assertRaises(ValueError):
            evaluate([])
        with self.assertRaises(ValueError):
            evaluate([row(gt=[[2, 0, 1, 2]])])
        with self.assertRaises(ValueError):
            evaluate([row(predictions=[prediction([0, 0, 5, 5], float("nan"))])])


if __name__ == "__main__":
    unittest.main()
