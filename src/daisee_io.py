import os

import pandas as pd


def clip_path(data_root: str, split_name: str, row) -> str:
    clip_id_ext = str(row["ClipID"]).strip()
    clip_id = clip_id_ext.replace(".avi", "").replace(".mp4", "")
    folder_id = clip_id[:6]
    return os.path.join(data_root, "DataSet", split_name, folder_id, clip_id, clip_id_ext)


def dominant_label(row) -> int:
    scores = {
        0: row["Boredom"],
        1: row["Confusion"],
        2: row["Engagement"],
        3: row["Frustration"],
    }
    return max(scores, key=scores.get)


def parse_daisee_csv(data_root, split_name):
    csv_path = os.path.join(data_root, "Labels", f"{split_name}Labels.csv")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    video_paths, labels = [], []

    for _, row in df.iterrows():
        video_path = clip_path(data_root, split_name, row)
        if os.path.exists(video_path):
            labels.append(dominant_label(row))
            video_paths.append(video_path)

    return video_paths, labels


def scan_split(data_root: str, split_name: str, max_missing_print: int = 20):
    csv_path = os.path.join(data_root, "Labels", f"{split_name}Labels.csv")
    if not os.path.isfile(csv_path):
        return {
            "split": split_name,
            "csv_path": csv_path,
            "error": "missing_csv",
            "csv_rows": 0,
            "found": 0,
            "missing": 0,
            "missing_examples": [],
            "label_counts": {0: 0, 1: 0, 2: 0, 3: 0},
        }

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    label_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    missing_examples = []

    found = 0
    for _, row in df.iterrows():
        p = clip_path(data_root, split_name, row)
        if os.path.exists(p):
            found += 1
            label_counts[dominant_label(row)] += 1
        else:
            if len(missing_examples) < max_missing_print:
                missing_examples.append((str(row["ClipID"]).strip(), p))

    n = len(df)
    return {
        "split": split_name,
        "csv_path": csv_path,
        "csv_rows": n,
        "found": found,
        "missing": n - found,
        "missing_examples": missing_examples,
        "label_counts": label_counts,
    }
