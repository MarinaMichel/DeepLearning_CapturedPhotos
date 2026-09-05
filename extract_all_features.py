"""
Step 2: Run MediaPipe feature extraction on every video listed in
wlasl_quarter.csv and save one .npy file per video_id.

Usage:
    python extract_all_features.py
    python extract_all_features.py --csv wlasl_quarter.csv --out extracted_features
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from extract_features import extract_features_from_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="wlasl_quarter.csv")
    parser.add_argument("--out", default="extracted_features")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    os.makedirs(args.out, exist_ok=True)

    skipped, done, failed = 0, 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_id = str(row["video_id"])
        out_path = os.path.join(args.out, f"{video_id}.npy")

        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            features = extract_features_from_video(row["local_path"])
            import numpy as np
            np.save(out_path, features)
            done += 1
        except Exception as e:
            print(f"Failed on {video_id}: {e}")
            failed += 1

    print(f"Done: {done} extracted, {skipped} already existed, {failed} failed")


if __name__ == "__main__":
    main()
