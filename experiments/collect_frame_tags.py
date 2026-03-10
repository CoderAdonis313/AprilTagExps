#!/usr/bin/env python3
"""
Interactive script to collect frame numbers and AprilTag IDs until 'q' is entered.

Saves entries to a CSV file (default: experiments/frame_tag_log.csv).
Each row: timestamp, frame, tags (semicolon-separated).

Usage:
  python experiments/collect_frame_tags.py --out experiments/frame_tag_log.csv
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys
from typing import List


def parse_tag_input(s: str) -> List[str]:
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect frame numbers and AprilTag IDs until 'q' to quit."
    )
    parser.add_argument(
        "--out",
        "-o",
        default=os.path.join("experiments", "frame_tag_log.csv"),
        help="Output CSV file path",
    )
    args = parser.parse_args()

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = ["timestamp", "frame", "tags"]
    file_exists = os.path.exists(out_path)

    try:
        with open(out_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)

            count = 0
            while True:
                try:
                    frame = input("Frame (or 'q' to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nInterrupted. Exiting.")
                    break

                if frame.lower() == "q":
                    break

                tags_in = input("AprilTag IDs (comma-separated, or leave empty): ").strip()
                if tags_in.lower() == "q":
                    break

                tags = parse_tag_input(tags_in)
                timestamp = datetime.datetime.now().isoformat()
                writer.writerow([timestamp, frame, ";".join(tags)])
                f.flush()
                count += 1
                print(f"Saved: frame={frame}, tags={tags}")

    except IOError as e:
        print(f"Failed to write to {out_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {count} entries to {out_path}")


if __name__ == "__main__":
    main()
