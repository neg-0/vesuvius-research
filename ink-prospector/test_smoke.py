"""Smoke tests for the Ink Prospector tool. No network. Run from this folder: python -m pytest -q
"""
import csv, os
import numpy as np
from PIL import Image
import textscore
import runs

SIZE = 256


def blank():
    # Dim, filled background: side() measures ink inside the valid area (the image minus a 0.35 mm edge), so an
    # all-zero image has no valid area and no ink at all.
    return np.full((SIZE, SIZE), 50, np.uint8)


def blobs(centres):
    # 60 px squares (about 2 mm at 34.56 um per px) are inside side()'s 0.3 to 6 mm structure range.
    img = blank()
    for (cy, cx) in centres:
        img[cy - 30:cy + 30, cx - 30:cx + 30] = 255
    return img


def save(path, arr):
    Image.fromarray(arr).save(path)


def test_score_sign_follows_face(tmp_path):
    # Face with several bright 20 px blobs versus a blank face.
    a = str(tmp_path / "a.png"); b = str(tmp_path / "b.png")
    save(a, blobs([(64, 64), (64, 192), (192, 64), (192, 192)]))
    save(b, blank())
    s, ta, tb = textscore.score(a, b, 8.64)
    assert s > 0, (s, ta, tb)  # positive when face a carries ink
    s2, _, _ = textscore.score(b, a, 8.64)
    assert s2 < 0, s2


def _write_patch(folder, pid, img_a, img_b):
    save(os.path.join(folder, f"p{pid:04d}_a.png"), img_a)
    save(os.path.join(folder, f"p{pid:04d}_b.png"), img_b)


def test_patches_best_first(tmp_path):
    folder = str(tmp_path / "scan")
    os.makedirs(folder)
    strong_a = blobs([(64, 64), (64, 192), (192, 64), (192, 192)])
    weak_a = blobs([(128, 128)])
    _write_patch(folder, 1, weak_a, blank())    # listed first in the file, weaker
    _write_patch(folder, 2, strong_a, blank())  # listed second, stronger
    with open(os.path.join(folder, "scores.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "seed_z", "seed_y", "seed_x", "status", "tracked", "area_cm2", "frac_a", "frac_b"])
        w.writerow([1, 10000, 3000, 5000, "ok", "0.9", "0.5", "0.20", "0.01"])
        w.writerow([2, 10100, 3000, 5100, "ok", "0.9", "0.5", "0.20", "0.01"])
    rows = runs.patches([folder], um=8.64)
    assert [r["id"] for r in rows] == [f"{os.path.basename(folder)}-2", f"{os.path.basename(folder)}-1"], [r["id"] for r in rows]
    assert rows[0]["ts"] >= rows[1]["ts"]
    assert rows[0]["side"] == "a" and rows[0]["seed"] == [10100, 3000, 5100]
