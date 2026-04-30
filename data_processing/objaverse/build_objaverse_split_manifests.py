from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


OBJAVERSE_ASSET_BASE_URL = "https://huggingface.co/datasets/allenai/objaverse/resolve/main"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build portable Objaverse train/val object manifests.")
    parser.add_argument("--train-prepared", default="data/objaverse_pp_task_relevant/prepared_quality_manifest.json")
    parser.add_argument("--val-prepared", default="data/objaverse_pp_task_relevant_val/prepared_quality_manifest.json")
    parser.add_argument("--out-dir", default="data_processing/objaverse/manifests")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = _portable_rows(Path(args.train_prepared), "train")
    val = _portable_rows(Path(args.val_prepared), "val")

    _write_json(out_dir / "objaversepp_task_relevant_train_objects.json", train)
    _write_json(out_dir / "objaversepp_task_relevant_val_objects.json", val)
    _write_json(
        out_dir / "objaversepp_task_relevant_splits.json",
        {
            "dataset": "objaversepp_task_relevant",
            "source": {
                "quality_annotations": "cindyxl/ObjaversePlusPlus annotated_800k.json",
                "asset_paths": "allenai/objaverse object-paths.json.gz",
                "categories": "allenai/objaverse lvis-annotations.json.gz",
            },
            "selection": {
                "min_score": 3,
                "styles": ["realistic", "scanned"],
                "densities": ["mid", "high"],
                "exclude": ["is_scene", "is_multi_object", "is_transparent", "is_single_color"],
                "category_preset": "driving,indoor",
                "points_per_object": 50000,
                "normalize_unit_box": True,
                "require_color_variance": True,
                "min_color_std": 3.0,
            },
            "splits": {
                "train": _summary(train),
                "val": _summary(val),
            },
            "objects": {
                "train": train,
                "val": val,
            },
        },
    )
    _write_csv(out_dir / "objaversepp_task_relevant_train_objects.csv", train)
    _write_csv(out_dir / "objaversepp_task_relevant_val_objects.csv", val)

    print(f"train objects: {len(train)}")
    print(f"val objects: {len(val)}")
    print(f"wrote manifests to: {out_dir}")


def _portable_rows(path: Path, split: str) -> list[dict[str, object]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    portable: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        asset_path = str(row["asset_path"])
        ply_file = Path(str(row["ply_path"])).name
        asset_file = Path(str(row["asset_local_path"])).name
        portable.append(
            {
                "split": split,
                "uid": row["uid"],
                "category": row["category"],
                "categories": row.get("categories", []),
                "score": row.get("score"),
                "style": row.get("style"),
                "density": row.get("density"),
                "asset_path": asset_path,
                "asset_url": f"{OBJAVERSE_ASSET_BASE_URL}/{asset_path}",
                "asset_file": asset_file,
                "ply_file": ply_file,
                "points": row.get("points"),
                "semantic_id": row.get("semantic_id"),
                "instance_id": row.get("instance_id"),
                "mean_rgb_channel_std": row.get("mean_rgb_channel_std"),
                "local_index": row.get("local_index"),
            }
        )
    portable.sort(key=lambda item: (str(item["category"]), int(item["local_index"]), str(item["uid"])))
    return portable


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    counts = Counter(str(row["category"]) for row in rows)
    return {
        "count": len(rows),
        "category_count": dict(sorted(counts.items())),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "split",
        "uid",
        "category",
        "score",
        "style",
        "density",
        "asset_path",
        "asset_url",
        "asset_file",
        "ply_file",
        "points",
        "semantic_id",
        "instance_id",
        "mean_rgb_channel_std",
        "local_index",
    ]
    lines = [",".join(columns)]
    for row in rows:
        values = [str(row.get(column, "")).replace('"', '""') for column in columns]
        lines.append(",".join(f'"{value}"' for value in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
