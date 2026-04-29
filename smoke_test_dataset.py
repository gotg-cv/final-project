#!/usr/bin/env python3
import os
import sys
import tempfile

import numpy as np
import torch


def write_minimal_mp4(path: str, num_frames: int = 64, fps: float = 8.0) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError("Could not open VideoWriter (codec/OS issue).")
    for i in range(num_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (i * 4 % 256, 64, 128)
        vw.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    vw.release()


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)

    tmp = tempfile.mkdtemp(prefix="daisee_smoke_")
    path = os.path.join(tmp, "clip.mp4")
    write_minimal_mp4(path, num_frames=64)

    from src.data_loader import DaiseeDataset

    ds = DaiseeDataset([path], [2])
    sample = ds[0]
    pv, y = sample["pixel_values"], sample["labels"]

    assert pv.shape == (16, 3, 224, 224), pv.shape
    assert y.dtype == torch.int64

    print("smoke_test_dataset: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
