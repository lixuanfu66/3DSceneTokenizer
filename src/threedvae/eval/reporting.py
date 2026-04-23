from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(slots=True)
class EvaluationReport:
    config: dict[str, object]
    dataset: dict[str, object]
    reconstruction: dict[str, object]
    codebook: dict[str, object]
    compression: dict[str, object]
    notes: list[str]


def write_evaluation_bundle(report: EvaluationReport, output_dir: str) -> dict[str, str]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    json_path = root / "evaluation_metrics.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(asdict(report), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    md_path = root / "evaluation_summary.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8", newline="\n")

    csv_path = root / "evaluation_metrics.csv"
    csv_path.write_text(_render_csv(report), encoding="utf-8", newline="\n")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
    }


def _render_markdown(report: EvaluationReport) -> str:
    lines = [
        "# 评估结果",
        "",
        "## 配置",
        "",
    ]
    for key, value in report.config.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 数据集", ""])
    for key, value in report.dataset.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 重建指标", ""])
    for key, value in report.reconstruction.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 码本指标", ""])
    for key, value in report.codebook.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 压缩指标", ""])
    for key, value in report.compression.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 备注", ""])
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _render_csv(report: EvaluationReport) -> str:
    rows = ["section,metric,value"]
    for section_name in ("config", "dataset", "reconstruction", "codebook", "compression"):
        section = getattr(report, section_name)
        for key, value in section.items():
            rows.append(f"{section_name},{key},{value}")
    return "\n".join(rows) + "\n"
