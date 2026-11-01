"""Paper metrics: COCO AP/PR summaries and a separate fixed-threshold Fa.

Rows contain original-image coordinates, including frames with no detections.
COCO matching runs in small batches; only matching arrays survive each batch.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
import io
import math

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRICS = ("mAP50", "mAP75", "mAP50_95", "Pr", "Rc", "F1", "Fa")


def summarize_envelope(precision, maximum_recall):
    """Use the exclusive PR101[:int(Rc * 100)] prefix in the CoMoE code."""
    precision = np.asarray(precision, dtype=np.float64)
    if (precision.shape != (101,) or not np.isfinite(precision).all()
            or np.any((precision < 0) | (precision > 1))):
        raise ValueError("Expected 101 finite precision values in [0, 1]")
    if maximum_recall is None:
        return {"Pr": None, "Rc": None, "F1": None, "precision_prefix_length": 0}
    recall = float(maximum_recall)
    if not math.isfinite(recall) or not 0 <= recall <= 1:
        raise ValueError("Recall must be in [0, 1]")
    count = int(recall * 100)
    if count == 0:
        # The upstream empty-prefix average is undefined, not zero precision.
        return {"Pr": None, "Rc": recall, "F1": None, "precision_prefix_length": 0}
    precision_mean = float(precision[:count].mean())
    return {"Pr": precision_mean, "Rc": recall,
            "F1": 2 * precision_mean * recall / (precision_mean + recall),
            "precision_prefix_length": count}


def _validate_row(row):
    if (not isinstance(row["domain"], str) or not row["domain"]
            or type(row["image_id"]) is not int or row["image_id"] < 1
            or any(type(row[k]) is not int or row[k] < 1 for k in ("width", "height"))):
        raise ValueError("Expected a domain name, positive integer image_id and image dimensions")
    for box in row["gt_boxes"]:
        if (len(box) != 4 or not all(math.isfinite(float(v)) for v in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("Ground-truth boxes must be finite xyxy boxes with positive area")
    if len(row["predictions"]) > 100:
        raise ValueError("The paper evaluation allows at most 100 detections per image")
    for detection in row["predictions"]:
        box, score = detection["bbox"], float(detection["score"])
        if (len(box) != 4 or not all(math.isfinite(float(v)) for v in box)
                or min(box[2:]) <= 0 or detection["category_id"] != 1
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Predictions require finite xywh boxes, score in [0, 1] and category_id=1")


def _xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [x1, y1, x2 - x1, y2 - y1]


def _iou(a, b):
    overlap = max(0., min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])) * max(
        0., min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    union = a[2] * a[3] + b[2] * b[3] - overlap
    return overlap / union if union > 0 else 0.


def _false_alarms(rows, threshold):
    """Historical score-descending greedy matching; equal IoUs select first GT."""
    true_positives = false_positives = 0
    for row in rows:
        targets = [_xywh(box) for box in row["gt_boxes"]]
        matched = set()
        for prediction in sorted(row["predictions"], key=lambda p: float(p["score"]), reverse=True):
            if float(prediction["score"]) < threshold:
                continue
            candidates = [(i, _iou(prediction["bbox"], target))
                          for i, target in enumerate(targets) if i not in matched]
            if candidates:
                index, overlap = max(candidates, key=lambda item: item[1])
                if overlap >= .5:
                    matched.add(index)
                    true_positives += 1
                    continue
            false_positives += 1
    return {"score_threshold": threshold, "matching_iou": .5,
            "true_positives": true_positives, "false_positives": false_positives,
            "Fa": false_positives / len(rows)}


def _parameters(evaluator, image_ids):
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.catIds = [1]
    # Keep COCO's original float64 0.50:0.05:0.95 grid and boundary semantics.
    evaluator.params.maxDets = [100]
    evaluator.params.areaRng = [evaluator.params.areaRng[0]]
    evaluator.params.areaRngLbl = ["all"]


def _match_batch(rows):
    images, annotations, detections = [], [], []
    for row in rows:
        image_id = row["image_id"]
        images.append({"id": image_id, "width": row["width"], "height": row["height"]})
        for box in row["gt_boxes"]:
            box = _xywh(box)
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": 1, "bbox": box, "area": box[2] * box[3],
                                "iscrowd": 0})
        detections.extend({"image_id": image_id, "bbox": list(d["bbox"]),
                           "score": float(d["score"]), "category_id": 1}
                          for d in row["predictions"])
    dataset = {"info": {}, "images": images, "annotations": annotations,
               "categories": [{"id": 1, "name": "target"}]}
    with redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = dataset
        truth.createIndex()
        if detections:
            predicted = truth.loadRes(detections)
        else:
            predicted = COCO()
            predicted.dataset = {**dataset, "annotations": []}
            predicted.createIndex()
        evaluator = COCOeval(truth, predicted, "bbox")
        _parameters(evaluator, [r["image_id"] for r in rows])
        evaluator.evaluate()
    # Accumulation only needs match/non-match, not the ground-truth numeric ID.
    return [None if item is None else {
        "dtScores": np.asarray(item["dtScores"], dtype=np.float64),
        "dtMatches": item["dtMatches"] > 0,
        "dtIgnore": item["dtIgnore"].astype(bool),
        "gtIgnore": item["gtIgnore"].astype(bool),
    } for item in evaluator.evalImgs]


def _domain_metrics(rows, batch_size, score_threshold):
    rows = sorted(rows, key=lambda r: r["image_id"])
    count = sum(len(row["gt_boxes"]) for row in rows)
    point = _false_alarms(rows, score_threshold)
    base = {"frames": len(rows), "ground_truth": count,
            "detections": sum(len(row["predictions"]) for row in rows),
            "Fa": point["Fa"], "fa_operating_point": point}
    if count == 0:
        return {**base, **{key: None for key in METRICS if key != "Fa"},
                "precision_prefix_length": 0, "PR101": None,
                "undefined_reason": "no_ground_truth"}
    evaluator = COCOeval(iouType="bbox")
    _parameters(evaluator, [row["image_id"] for row in rows])
    evaluator.evalImgs = []
    for offset in range(0, len(rows), batch_size):
        evaluator.evalImgs.extend(_match_batch(rows[offset:offset + batch_size]))
    evaluator._paramsEval = deepcopy(evaluator.params)
    with redirect_stdout(io.StringIO()):
        evaluator.accumulate()
    precision = evaluator.eval["precision"][:, :, 0, 0, 0]
    recall = float(evaluator.eval["recall"][0, 0, 0, 0])
    if np.any(precision < 0) or not np.isfinite(precision).all():
        raise ValueError("Undefined COCO precision despite nonempty ground truth")
    summary = summarize_envelope(precision[0], recall)
    return {**base, "mAP50": float(precision[0].mean()),
            "mAP75": float(precision[5].mean()), "mAP50_95": float(precision.mean()),
            **summary, "PR101": precision[0].tolist(),
            "undefined_reason": None if summary["Pr"] is not None else "empty_precision_prefix"}


def evaluate(rows, *, batch_size=128, score_threshold=.5):
    """Evaluate complete domains, preserving empty frames and image-ID tie order.

    Each row has image_id, domain, width, height, gt_boxes (xyxy) and predictions
    (bbox in xywh, score and category_id=1). AP, Pr, Rc and F1 are fractions;
    Fa is false detections per image. Macro metrics are equal domain means,
    including mean domain F1. An undefined domain makes that macro field null.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 1024:
        raise ValueError("batch_size must be an integer from 1 to 1024")
    if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
        raise ValueError("score_threshold must be in [0, 1]")
    domains, identities = defaultdict(list), set()
    for row in rows:
        _validate_row(row)
        identity = (row["domain"], row["image_id"])
        if identity in identities:
            raise ValueError("Duplicate domain/image_id")
        identities.add(identity)
        domains[row["domain"]].append(row)
    if not domains:
        raise ValueError("No evaluation frames")
    per_domain = {name: _domain_metrics(values, batch_size, score_threshold)
                  for name, values in sorted(domains.items())}
    macro = {}
    for name in METRICS:
        values = [result[name] for result in per_domain.values()]
        macro[name] = None if any(value is None for value in values) else float(np.mean(values))
    return {"per_domain": per_domain, "macro": macro, "definitions": {
        "units": "AP, Pr, Rc and F1 are fractions; Fa is false positives per frame",
        "AP": "COCO 101-point interpolated AP, maxDet=100, all areas",
        "Pr": "mean(PR101[:int(Rc*100)]) at IoU=0.5",
        "Rc": "COCO maximum recall at IoU=0.5",
        "F1": "2*Pr*Rc/(Pr+Rc); not a fixed-confidence or best-F1 result",
        "Fa": f"score>={score_threshold:g}, IoU>=0.5; descending-score greedy matching",
        "macro": "equal mean of domain metrics; F1 is the mean of domain F1 values",
        "undefined": "null; undefined domains are not silently dropped from macro averages",
    }}
