"""Causal boundaries, archived target geometry and portable annotation conversion."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from kfgmoe.data import CausalDataset, causal_paths, collate, read_annotations
from tools.prepare_annotations import convert


class DataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="kfgmoe_data_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.video = self.root / "DAUB-H" / "video1"
        self.video.mkdir(parents=True)
        Image.new("RGB", (77, 83), (91, 91, 91)).save(self.video / "0.bmp")

    def manifest(self, boxes):
        path = self.root / "annotations.jsonl"
        path.write_text(json.dumps({"path": "DAUB-H/video1/0.bmp", "domain": 1,
                                   "boxes": boxes}) + "\n", encoding="utf-8")
        return read_annotations(path, self.root)

    def test_first_frame_repeats_without_future_access(self):
        paths = causal_paths(self.video / "0.bmp", 5)
        self.assertEqual(paths, [self.video / "0.bmp"] * 5)
        images, _, _ = CausalDataset(self.manifest([]), 64)[0]
        for i in range(1, 5):
            self.assertTrue(torch.equal(images[:, 0], images[:, i]))

    def test_missing_history_is_rejected_instead_of_crossing_videos(self):
        Image.new("RGB", (77, 83)).save(self.video / "2.bmp")
        another_video = self.video.with_name("video2")
        another_video.mkdir()
        Image.new("RGB", (77, 83)).save(another_video / "1.bmp")
        with self.assertRaises(FileNotFoundError):
            causal_paths(self.video / "2.bmp", 5)

    def test_integer_letterbox_target_geometry(self):
        records = self.manifest([[3, 5, 19, 33, 0]])
        dataset = CausalDataset(records, 64)
        image, targets, domain = dataset[0]
        # 77x83 -> 59x64, left padding 2; native integer assignments
        # produce xyxy=(4,3,16,25), then cxcywh=(10,14,12,22).
        self.assertEqual(tuple(image.shape), (3, 5, 64, 64))
        self.assertTrue(torch.equal(targets, torch.tensor([[10., 14., 12., 22., 0.]])))
        self.assertEqual(domain, 1)
        self.assertTrue(np.array_equal(records[0]["boxes"], [[3, 5, 19, 33, 0]]))
        self.assertTrue(torch.equal(dataset[0][1], targets))

    def test_empty_annotation_and_batch_one(self):
        value = CausalDataset(self.manifest([]), 64)[0]
        self.assertEqual(tuple(value[1].shape), (0, 5))
        images, boxes, domains = collate([value])
        self.assertEqual(tuple(images.shape), (1, 3, 5, 64, 64))
        self.assertEqual(tuple(boxes[0].shape), (0, 5))
        self.assertEqual(domains.tolist(), [1])

    def test_portable_conversion_preserves_boxes_and_empty_frames(self):
        source, output = self.root / "original.txt", self.root / "portable.jsonl"
        source.write_text("/old/root/DAUB-H/video1/0.bmp 3,5,19,33,0\n"
                          "/old/root/DAUB-H/video1/1.bmp\n", encoding="utf-8")
        self.assertEqual(convert(source, output, self.root, "/old/root"), 2)
        records = read_annotations(output, self.root)
        self.assertEqual(records[0]["key"], "DAUB-H/video1/0.bmp")
        self.assertEqual(records[0]["domain"], 1)
        self.assertTrue(np.array_equal(records[0]["boxes"], [[3, 5, 19, 33, 0]]))
        self.assertEqual(records[1]["boxes"].shape, (0, 5))
        with self.assertRaises(FileExistsError):
            convert(source, output, self.root, "/old/root")


if __name__ == "__main__":
    unittest.main()
