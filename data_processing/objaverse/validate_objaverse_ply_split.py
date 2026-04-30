from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from threedvae.data.loaders.ply_loader import load_ply_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a prepared Objaverse PLY split.")
    parser.add_argument("--ply-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-points", type=int, default=50000)
    args = parser.parse_args()

    ply_dir = Path(args.ply_dir)
    manifest_path = Path(args.manifest)
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("objects", [])

    category_counts: Counter[str] = Counter()
    missing: list[str] = []
    bad: list[str] = []
    for row in rows:
        file_name = str(row.get("ply_file") or Path(str(row.get("ply_path", ""))).name)
        category = str(row.get("category", "unknown"))
        path = ply_dir / file_name
        if not path.exists():
            missing.append(file_name)
            continue
        frame = load_ply_frame(path)
        if frame.xyz.shape[0] != args.expected_points:
            bad.append(f"{file_name}: expected {args.expected_points} points, got {frame.xyz.shape[0]}")
        if frame.rgb.shape[0] != frame.xyz.shape[0]:
            bad.append(f"{file_name}: rgb count mismatch")
        if frame.semantic_id.shape[0] != frame.xyz.shape[0] or frame.instance_id.shape[0] != frame.xyz.shape[0]:
            bad.append(f"{file_name}: label count mismatch")
        category_counts[category] += 1

    print(f"manifest: {manifest_path}")
    print(f"ply_dir: {ply_dir}")
    print(f"objects: {len(rows)}")
    print(f"missing: {len(missing)}")
    print(f"bad: {len(bad)}")
    print("categories:")
    for category, count in category_counts.most_common():
        print(f"  {category}: {count}")

    if missing:
        print("missing files:")
        for item in missing:
            print(f"  {item}")
    if bad:
        print("bad files:")
        for item in bad:
            print(f"  {item}")
    if missing or bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
