import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from callability_signal_experiment import (  # noqa: E402
    add_loso_context_features,
    aggregate_merged_span_comparison,
    classification_metrics,
    loso_classifier,
    merge_role_sequence,
    merged_span_metrics,
    normalize_role,
    plot_confusion,
    write_json,
    write_jsonl,
)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_signal_rows(bars_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(bars_dir.glob("*/*.signal_bars.jsonl")):
        song_id = path.parent.name
        for row in load_jsonl(path):
            item = dict(row)
            item["_song_id"] = str(item.get("song_id") or song_id)
            rows.append(item)
    rows.sort(key=lambda item: (str(item["_song_id"]), int(item["bar_index"])))
    return rows


def group_by_song(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["_song_id"]), []).append(row)
    for song_rows in grouped.values():
        song_rows.sort(key=lambda item: int(item["bar_index"]))
    return grouped


def aggregate_search_merged_metrics(rows: Sequence[Dict[str, Any]], predictions: Sequence[Dict[str, Any]], method: str) -> Dict[str, Any]:
    prediction_map = {
        (str(item["song_id"]), int(item["bar_index"])): normalize_role(item["prediction"])
        for item in predictions
    }
    fake_results = []
    for song_id, song_rows in group_by_song(rows).items():
        target_roles = [normalize_role(row["target"]["call_role"]) for row in song_rows]
        pred_roles = [
            prediction_map.get((song_id, int(row["bar_index"])), "keepspace")
            for row in song_rows
        ]
        target_merged = merge_role_sequence(song_rows, target_roles, method="target_manual_grid")
        predicted_merged = merge_role_sequence(song_rows, pred_roles, method=method)
        full_bar_durations = [
            float(row["features"]["duration"])
            for row in song_rows
            if row.get("bar_kind") == "full_bar"
        ]
        tolerance_seconds = float(np.median(full_bar_durations)) if full_bar_durations else 2.5
        comparison = merged_span_metrics(target_merged, predicted_merged, tolerance_seconds)
        fake_results.append({"metrics": {"merged_span_comparison": {method: comparison}}})
    return aggregate_merged_span_comparison(fake_results).get(method, {})


def candidate_grid() -> List[Dict[str, Any]]:
    model_kinds = [
        "vote_rf1_logreg1_gb1",
        "vote_rf2_logreg1_gb1",
        "vote_rf3_logreg1_gb1",
    ]
    return [
        {
            "method": f"search_audio_{model_kind}",
            "include_audio": True,
            "include_context": False,
            "model_kind": model_kind,
        }
        for model_kind in model_kinds
    ]


def summary_lines(results: Sequence[Dict[str, Any]]) -> List[str]:
    ordered = sorted(
        results,
        key=lambda item: (
            float(item["classification"]["macro_f1"]),
            float(item["merged"].get("macro_role_iou", 0.0)),
            float(item["merged"].get("boundary", {}).get("f1", 0.0)),
        ),
        reverse=True,
    )
    lines = [
        "# Call-Role Model Search",
        "",
        "| Method | Accuracy | Macro-F1 | Time-W Acc | Macro IoU | Span Boundary F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in ordered:
        cls = item["classification"]
        merged = item["merged"]
        lines.append(
            f"| {item['method']} | {cls['accuracy']:.3f} | {cls['macro_f1']:.3f} | "
            f"{merged.get('time_weighted_role_accuracy', 0.0):.3f} | "
            f"{merged.get('macro_role_iou', 0.0):.3f} | "
            f"{merged.get('boundary', {}).get('f1', 0.0):.3f} |"
        )
    if ordered:
        best = ordered[0]
        lines.extend(
            [
                "",
                f"Best by macro-F1: `{best['method']}` "
                f"(accuracy={best['classification']['accuracy']:.3f}, "
                f"macro-F1={best['classification']['macro_f1']:.3f}).",
            ]
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Search call-role models from saved bar-level signal features.")
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/signal_callability_model_search/round1"))
    args = parser.parse_args()

    rows = load_signal_rows(args.bars_dir)
    if len({row["_song_id"] for row in rows}) < 2:
        raise SystemExit("Need at least two songs for leave-one-song-out search.")
    add_loso_context_features(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_predictions = []
    for candidate in candidate_grid():
        method = candidate["method"]
        print(f"Searching {method}", flush=True)
        classification, predictions = loso_classifier(
            rows,
            method=method,
            include_audio=bool(candidate.get("include_audio")),
            include_context=bool(candidate.get("include_context")),
            model_kind=str(candidate.get("model_kind")),
            decoder=candidate.get("decoder"),
            transition_weight=float(candidate.get("transition_weight", 0.7)),
            boundary_bonus=float(candidate.get("boundary_bonus", 0.55)),
        )
        merged = aggregate_search_merged_metrics(rows, predictions, method)
        result = {
            "method": method,
            "candidate": candidate,
            "classification": classification,
            "merged": merged,
        }
        results.append(result)
        all_predictions.extend(predictions)
        partial_results = sorted(results, key=lambda item: float(item["classification"]["macro_f1"]), reverse=True)
        write_json(args.out_dir / "model_search_metrics.json", {"results": partial_results})
        write_jsonl(args.out_dir / "model_search_predictions.jsonl", all_predictions)
        (args.out_dir / "model_search_summary.md").write_text(
            "\n".join(summary_lines(partial_results)) + "\n",
            encoding="utf-8",
        )

    results.sort(key=lambda item: float(item["classification"]["macro_f1"]), reverse=True)
    write_json(args.out_dir / "model_search_metrics.json", {"results": results})
    write_jsonl(args.out_dir / "model_search_predictions.jsonl", all_predictions)
    (args.out_dir / "model_search_summary.md").write_text("\n".join(summary_lines(results)) + "\n", encoding="utf-8")
    if results:
        plot_confusion(
            results[0]["classification"]["confusion"],
            args.out_dir / "best_model_confusion.png",
            f"{results[0]['method']} confusion",
        )
    print(f"Wrote search summary: {args.out_dir / 'model_search_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
