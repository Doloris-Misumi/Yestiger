import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

from callability_signal_experiment import load_json, write_json


TARGETS = ("manual_fine", "manual_coarse")
METRIC_FILENAMES = (
    "external_structure_metrics.json",
    "official_msaf_metrics.json",
    "mert_structure_metrics.json",
)


def baseline_rows(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    rows = []
    comparison = payload.get("music_boundary_comparison") or {}
    for target_name in TARGETS:
        for method_name in ("allin1_structure", "fused_novelty_topk"):
            metrics = (comparison.get(target_name) or {}).get(method_name)
            if not metrics:
                continue
            rows.append(
                {
                    "target": target_name,
                    "group": "00_previous_baseline",
                    "method": method_name,
                    "uses_allin1": method_name == "allin1_structure",
                    "uses_manual_training": False,
                    "aggregate": metrics,
                    "source": str(path.parent),
                }
            )
    return rows


def load_result_rows(round_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    rows = []
    seen = set()
    for round_dir in round_dirs:
        for filename in METRIC_FILENAMES:
            path = round_dir / filename
            if not path.exists():
                continue
            payload = load_json(path)
            for item in payload.get("results") or []:
                key = (item.get("target"), item.get("method"))
                if key in seen:
                    continue
                seen.add(key)
                row = dict(item)
                row["source"] = str(round_dir)
                rows.append(row)
    return rows


def load_best_hybrid(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    wanted = {
        ("manual_fine", "allin1_plus_learned_rf_scale0p65"),
        ("manual_coarse", "allin1_structure"),
    }
    rows = []
    for item in payload.get("results") or []:
        key = (item.get("target"), item.get("method"))
        if key not in wanted:
            continue
        row = dict(item)
        row["group"] = "05_hybrid_allin1_plus_signal" if "allin1_plus" in str(item.get("method")) else "00_previous_baseline"
        row["uses_allin1"] = True
        row["uses_manual_training"] = "learned" in str(item.get("method"))
        row["source"] = str(path.parent)
        rows.append(row)
    return rows


def format_row(item: Dict[str, Any]) -> str:
    metrics = item["aggregate"]
    return (
        f"| {item['target']} | {item.get('group', '')} | {item['method']} | "
        f"{str(bool(item.get('uses_allin1', False))).lower()} | "
        f"{str(bool(item.get('uses_manual_training', False))).lower()} | "
        f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | "
        f"{metrics['target_count']} | {metrics['predicted_count']} | {item.get('source', '')} |"
    )


def summary_lines(rows: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "# Structure Segmentation All Methods Summary",
        "",
        "| Target | Group | Method | Uses allin1 | Uses manual training | Precision | Recall | F1 | Target Bnd | Pred Bnd | Source |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(rows, key=lambda item: (item["target"], -float(item["aggregate"]["f1"]), str(item.get("group", ""))))
    lines.extend(format_row(item) for item in ordered)
    lines.extend(["", "## Main Comparisons", ""])
    for target_name in TARGETS:
        subset = [item for item in rows if item["target"] == target_name]
        non_allin1 = [item for item in subset if not bool(item.get("uses_allin1", False))]
        overall = max(subset, key=lambda item: float(item["aggregate"]["f1"]))
        non_allin1_best = max(non_allin1, key=lambda item: float(item["aggregate"]["f1"]))
        lines.append(
            f"- `{target_name}` overall best: `{overall['method']}` F1={overall['aggregate']['f1']:.3f}; "
            f"best non-allin1: `{non_allin1_best['method']}` F1={non_allin1_best['aggregate']['f1']:.3f}."
        )
    lines.extend(
        [
            "",
            "## Method Notes",
            "",
            "- `official_msaf_foote_bar_features_loso_tuned` uses the official MSAF Foote Segmenter module on existing bar-level features. Full `msaf.process` timed out on full MP3 files, and CNMF import failed because `cvxopt` could not load a DLL.",
            "- `mert95m_contextual_foote_loso_tuned` uses MERT-v1-95M weights loaded as a standard Wav2Vec2Model with exact state-dict match. No Hugging Face remote code was executed.",
            "- MERT embeddings were cached per song under `experiments/mert_structure_segmentation/cache/`.",
        ]
    )
    return lines


def save_code_snapshot(out_dir: Path) -> None:
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        Path("scripts/aggregate_structure_segmentation_all.py"),
        Path("scripts/external_structure_segmentation.py"),
        Path("scripts/official_msaf_segmentation.py"),
        Path("scripts/mert_embedding_segmentation.py"),
    ):
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate all structure segmentation method results.")
    parser.add_argument("--baseline-metrics", type=Path, default=Path("experiments/signal_callability/aggregate_signal_metrics.json"))
    parser.add_argument("--round-dir", type=Path, action="append", required=True)
    parser.add_argument("--hybrid-metrics", type=Path, default=Path("experiments/signal_callability_music_boundary_search/round3_allin1_plus_smallscale/music_boundary_search_metrics.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/structure_segmentation_all_methods/final"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = baseline_rows(args.baseline_metrics) + load_result_rows(args.round_dir) + load_best_hybrid(args.hybrid_metrics)
    deduped = []
    seen = set()
    for row in rows:
        key = (row.get("target"), row.get("method"), row.get("group"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    write_json(args.out_dir / "structure_segmentation_all_metrics.json", {"results": deduped})
    (args.out_dir / "STRUCTURE_SEGMENTATION_ALL_METHODS.md").write_text(
        "\n".join(summary_lines(deduped)) + "\n",
        encoding="utf-8",
    )
    save_code_snapshot(args.out_dir)
    print(f"Wrote summary: {args.out_dir / 'STRUCTURE_SEGMENTATION_ALL_METHODS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
