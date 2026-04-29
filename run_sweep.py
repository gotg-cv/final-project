#!/usr/bin/env python3
"""
Run several training configs sequentially (cheap hyperparameter search).
Reads sweep_presets.json: list of overrides merged into config.json.

Example presets entry: {"tag": "lr1e4", "overrides": {"learning_rate": 0.0001}, "epochs": 1}
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--base_config", default="config.json")
    ap.add_argument("--sweep_file", default="sweep_presets.json")
    ap.add_argument("--output_parent", default="outputs/sweeps")
    ap.add_argument("--max_train_samples", type=int, default=None, help="Cap all runs for quick search")
    ap.add_argument("--max_eval_samples", type=int, default=None)
    args = ap.parse_args()

    with open(args.base_config) as f:
        base = json.load(f)
    with open(args.sweep_file) as f:
        spec = json.load(f)

    os.makedirs(args.output_parent, exist_ok=True)
    results = []
    exe = sys.executable
    root = os.path.dirname(os.path.abspath(__file__))

    for i, run in enumerate(spec.get("runs", [])):
        tag = run.get("tag", f"run{i}")
        cfg = copy.deepcopy(base)
        for k, v in run.get("overrides", {}).items():
            cfg[k] = v
        if "epochs" in run:
            cfg["num_train_epochs"] = run["epochs"]
        out_cfg = os.path.join(args.output_parent, f"config_{tag}.json")
        with open(out_cfg, "w") as wf:
            json.dump(cfg, wf, indent=2)

        out_dir = os.path.join(args.output_parent, tag)
        cmd = [
            exe,
            os.path.join(root, "src", "train.py"),
            "--data_root",
            args.data_root,
            "--config",
            out_cfg,
            "--output_dir",
            out_dir,
        ]
        if args.max_train_samples:
            cmd += ["--max_train_samples", str(args.max_train_samples)]
        if args.max_eval_samples:
            cmd += ["--max_eval_samples", str(args.max_eval_samples)]

        print("+", " ".join(cmd))
        subprocess.run(cmd, cwd=root, check=True)

        eval_cmd = [
            exe,
            os.path.join(root, "src", "evaluate.py"),
            "--data_root",
            args.data_root,
            "--model_path",
            os.path.join(out_dir, "final"),
            "--out_dir",
            os.path.join(out_dir, "eval_report"),
        ]
        if args.max_eval_samples:
            eval_cmd += ["--max_samples", str(args.max_eval_samples)]
        print("+", " ".join(eval_cmd))
        subprocess.run(eval_cmd, cwd=root, check=False)

        mpath = os.path.join(out_dir, "eval_report", "metrics.json")
        row = {"tag": tag, "metrics_path": mpath}
        if os.path.isfile(mpath):
            with open(mpath) as mf:
                data = json.load(mf)
            row["macro_f1"] = data.get("macro_f1")
        results.append(row)

    summary = os.path.join(args.output_parent, "sweep_summary.json")
    with open(summary, "w") as wf:
        json.dump(results, wf, indent=2)
    print("Wrote", summary)


if __name__ == "__main__":
    main()
