#!/usr/bin/env python3
import argparse
import os
import sys

from src.daisee_io import scan_split


def report(data_root: str, split: str, max_missing: int):
    s = scan_split(data_root, split, max_missing_print=max_missing)
    if s.get("error") == "missing_csv":
        print(f"{split}: no file {s['csv_path']}")
        return 0, 0

    print(f"{split}: {s['csv_rows']} rows in CSV, {s['found']} files found, {s['missing']} missing")
    lc = s["label_counts"]
    print(f"  label counts (Boredom, Confusion, Engagement, Frustration): {lc[0]} {lc[1]} {lc[2]} {lc[3]}")
    if s["missing_examples"]:
        print("  example missing:")
        for clip_id, expected_path in s["missing_examples"]:
            print(f"    ClipID {clip_id} -> {expected_path}")
        if s["missing"] > len(s["missing_examples"]):
            print(f"    … ({s['missing'] - len(s['missing_examples'])} more missing)")
    return s["found"], s["csv_rows"]


def main() -> int:
    p = argparse.ArgumentParser(description="Verify DAiSEE paths against Labels CSV")
    p.add_argument("--data_root", required=True, help="DAiSEE root (contains Labels/ and DataSet/)")
    p.add_argument("--max_missing", type=int, default=20, help="cap on printed missing ClipID paths")
    a = p.parse_args()
    if not os.path.isdir(a.data_root):
        print(f"Not a directory: {a.data_root}")
        return 1

    tr_ok, _ = report(a.data_root, "Train", a.max_missing)
    va_ok, _ = report(a.data_root, "Validation", a.max_missing)

    if tr_ok == 0 or va_ok == 0:
        print("\nFix paths or unzip data_root before training (train script skips missing files).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
