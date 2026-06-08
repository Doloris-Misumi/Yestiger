import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from callability_signal_experiment import load_json, write_json


TARGETS = ("manual_fine", "manual_coarse")


def load_results(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    seen = set()
    for path in paths:
        metrics_path = path / "external_structure_metrics.json"
        if not metrics_path.exists():
            continue
        payload = load_json(metrics_path)
        for item in payload.get("results") or []:
            key = (item.get("target"), item.get("method"))
            if key in seen:
                continue
            seen.add(key)
            copied = dict(item)
            copied["source_round"] = str(path)
            results.append(copied)
    return results


def baseline_rows(baseline_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    comparison = baseline_metrics.get("music_boundary_comparison") or {}
    for target_name in TARGETS:
        for method_name in ("allin1_structure", "fused_novelty_topk"):
            metrics = (comparison.get(target_name) or {}).get(method_name)
            if not metrics:
                continue
            rows.append(
                {
                    "target": target_name,
                    "method": method_name,
                    "group": "00_previous_baseline",
                    "uses_allin1": method_name == "allin1_structure",
                    "uses_manual_training": False,
                    "aggregate": metrics,
                    "source_round": "experiments/signal_callability",
                }
            )
    return rows


def metric_line(item: Dict[str, Any]) -> str:
    metrics = item["aggregate"]
    return (
        f"| {item['target']} | {item['group']} | {item['method']} | "
        f"{str(bool(item.get('uses_allin1', False))).lower()} | "
        f"{str(bool(item.get('uses_manual_training', False))).lower()} | "
        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | "
        f"{metrics['target_count']} | {metrics['predicted_count']} | {item['source_round']} |"
    )


def summary_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "# External Structure Segmentation Final Summary",
        "",
        "| Target | Group | Method | Uses allin1 | Uses manual training | Precision | Recall | F1 | Target Bnd | Pred Bnd | Source |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(rows, key=lambda item: (item["target"], -float(item["aggregate"]["f1"]), str(item["group"]), str(item["method"])))
    lines.extend(metric_line(item) for item in ordered)
    lines.extend(["", "## Takeaways", ""])
    for target_name in TARGETS:
        subset = [item for item in rows if item["target"] == target_name]
        non_allin1 = [item for item in subset if not bool(item.get("uses_allin1", False))]
        pure_audio_best = max(non_allin1, key=lambda item: float(item["aggregate"]["f1"]))
        overall_best = max(subset, key=lambda item: float(item["aggregate"]["f1"]))
        lines.append(
            f"- `{target_name}` overall best: `{overall_best['method']}` F1={overall_best['aggregate']['f1']:.3f}; "
            f"best non-allin1: `{pure_audio_best['method']}` F1={pure_audio_best['aggregate']['f1']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Code And Result Locations",
            "",
            "- Main implementation: `scripts/external_structure_segmentation.py`.",
            "- Aggregation script: `scripts/aggregate_external_structure_results.py`.",
            "- Each round contains a `code_snapshot/` directory and per-method `methods/<group>/<method>/<target>/` metrics/predictions.",
        ]
    )
    return lines


def save_code_snapshot(out_dir: Path) -> None:
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        Path("scripts/external_structure_segmentation.py"),
        Path("scripts/aggregate_external_structure_results.py"),
    ):
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate external structure segmentation rounds.")
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--baseline-metrics", type=Path, default=Path("experiments/signal_callability/aggregate_signal_metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/external_structure_segmentation/final_summary"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = baseline_rows(load_json(args.baseline_metrics)) + load_results(args.round_dir)
    write_json(args.out_dir / "external_structure_final_metrics.json", {"results": rows})
    (args.out_dir / "EXTERNAL_STRUCTURE_SEGMENTATION_LOG.md").write_text(
        "\n".join(summary_lines(rows)) + "\n",
        encoding="utf-8",
    )
    save_code_snapshot(args.out_dir)
    print(f"Wrote final summary: {args.out_dir / 'EXTERNAL_STRUCTURE_SEGMENTATION_LOG.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
