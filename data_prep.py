"""
Step 1: Download WLASL metadata + videos, then cut it down to a quarter
of the sign classes (glosses) so the rest of the pipeline runs on a
smaller, faster subset.

Usage:
    python data_prep.py
    python data_prep.py --fraction 0.25 --mode top      # default: keep the
                                                          # 25% most-represented
                                                          # classes (best for
                                                          # training quality)
    python data_prep.py --fraction 0.25 --mode random    # or a random 25%
                                                          # of classes instead

Output:
    wlasl_quarter.csv  -- one row per video instance that belongs to the
                           kept classes AND whose video file exists locally.
"""
import argparse
import json
import os

import kagglehub
import pandas as pd


def load_wlasl_metadata(json_path: str) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        wlasl_data = json.load(f)

    records = []
    for entry in wlasl_data:
        gloss = entry.get("gloss")
        for inst in entry.get("instances", []):
            records.append({
                "gloss": gloss,
                "video_id": inst.get("video_id"),
                "fps": inst.get("fps"),
                "split": inst.get("split"),
                "url": inst.get("url"),
            })
    return pd.DataFrame(records)


def select_quarter_classes(df: pd.DataFrame, fraction: float = 0.25, mode: str = "top") -> pd.DataFrame:
    """Keep only `fraction` of the distinct glosses (classes), not rows."""
    counts = df["gloss"].value_counts()
    n_keep = max(1, int(len(counts) * fraction))

    if mode == "top":
        # Keep the classes with the most video instances -> more training
        # data per class, generally gives a stronger/more stable model.
        keep_glosses = counts.head(n_keep).index.tolist()
    elif mode == "random":
        keep_glosses = counts.sample(n=n_keep, random_state=42).index.tolist()
    else:
        raise ValueError("mode must be 'top' or 'random'")

    return df[df["gloss"].isin(keep_glosses)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=0.25,
                         help="Fraction of classes to keep (default 0.25 = a quarter)")
    parser.add_argument("--mode", choices=["top", "random"], default="top",
                         help="'top' keeps the most-represented classes, "
                              "'random' picks classes at random")
    parser.add_argument("--out", default="wlasl_quarter.csv")
    args = parser.parse_args()

    path = kagglehub.dataset_download("risangbaskoro/wlasl-processed")
    print("Dataset downloaded to:", path)

    json_file_path = os.path.join(path, "WLASL_v0.3.json")
    df = load_wlasl_metadata(json_file_path)
    print(f"Full dataset: {len(df)} instances across {df['gloss'].nunique()} classes")

    df_quarter = select_quarter_classes(df, fraction=args.fraction, mode=args.mode)
    print(f"Kept {df_quarter['gloss'].nunique()} classes "
          f"({args.mode}, fraction={args.fraction}) -> {len(df_quarter)} instances")

    videos_dir = os.path.join(path, "videos")
    df_quarter["local_path"] = df_quarter["video_id"].astype(str).apply(
        lambda vid: os.path.join(videos_dir, f"{vid}.mp4")
    )
    df_quarter["file_exists"] = df_quarter["local_path"].apply(os.path.exists)

    df_final = df_quarter[df_quarter["file_exists"]].reset_index(drop=True)
    print(f"Videos actually present on disk: {len(df_final)}")

    df_final.to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
