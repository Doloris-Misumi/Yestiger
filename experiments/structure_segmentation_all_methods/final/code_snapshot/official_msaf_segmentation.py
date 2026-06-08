import argparse
import copy
import importlib
import json
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from callability_signal_experiment import boundary_detection_metrics_seconds, load_json, write_json
from external_structure_segmentation import bar_feature_matrix, indices_to_times
from search_music_boundary_detectors import load_song_records, target_for


TARGETS = ("manual_fine", "manual_coarse")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def patch_msaf_imports() -> None:
    """Let MSAF import on this Windows/Python 3.9 environment.

    MSAF imports every algorithm at package import time. Its CNMF path imports
    cvxopt, which fails to load a DLL here. The other official algorithm
    modules are still useful, so we skip only CNMF and keep the rest official.
    """
    real_import_module = importlib.import_module

    def patched_import_module(name: str, package: Any = None) -> Any:
        if name.endswith(".cnmf") or name == "msaf.algorithms.cnmf":
            module = types.ModuleType(name)
            module.is_boundary_type = False
            module.is_label_type = False
            sys.modules[name] = module
            return module
        return real_import_module(name, package)

    importlib.import_module = patched_import_module


def import_msaf_modules() -> Dict[str, Any]:
    patch_msaf_imports()
    from scipy import signal

    if not hasattr(signal, "gaussian"):
        signal.gaussian = signal.windows.gaussian

    from msaf.algorithms import foote, sf

    return {"foote": foote, "sf": sf}


@dataclass
class FakeFileStruct:
    audio_file: str


@dataclass
class FakeFeatures:
    features: np.ndarray


def candidate_indices_from_msaf(est_idxs: Sequence[int], n_bars: int) -> List[int]:
    output = []
    for value in est_idxs:
        index = int(value)
        if 0 < index < n_bars - 1:
            output.append(index)
    return sorted(set(output))


def run_msaf_candidate(record: Dict[str, Any], candidate: Dict[str, Any], modules: Dict[str, Any]) -> List[int]:
    module = modules[str(candidate["algorithm"])]
    features = bar_feature_matrix(record, str(candidate["feature_set"]))
    n_bars = features.shape[0]
    if n_bars <= 3:
        return []
    config = copy.deepcopy(candidate["config"])
    segmenter = module.Segmenter(
        FakeFileStruct(str(record["song_id"])),
        feature="mfcc",
        annot_beats=False,
        framesync=False,
        features=FakeFeatures(features),
        **config,
    )
    est_idxs, _est_labels = segmenter.processFlat()
    return candidate_indices_from_msaf(est_idxs, n_bars=n_bars)


def aggregate_metrics(per_song: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(item["tp"]) for item in per_song)
    fp = sum(int(item["fp"]) for item in per_song)
    fn = sum(int(item["fn"]) for item in per_song)
    target_count = sum(int(item["target_count"]) for item in per_song)
    predicted_count = sum(int(item["predicted_count"]) for item in per_song)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "target_count": target_count,
        "predicted_count": predicted_count,
        "mean_tolerance_seconds": round(float(np.mean([float(item["tolerance_seconds"]) for item in per_song])), 6) if per_song else 0.0,
    }


def evaluate_fixed(
    records: Sequence[Dict[str, Any]],
    candidate: Dict[str, Any],
    target_name: str,
    modules: Dict[str, Any],
) -> Dict[str, Any]:
    per_song = []
    predictions = []
    for record in records:
        boundary_indices = run_msaf_candidate(record, candidate, modules)
        predicted = indices_to_times(record, boundary_indices)
        target = target_for(record, target_name)
        metrics = boundary_detection_metrics_seconds(target, predicted, float(record["tolerance_seconds"]))
        metrics["song_id"] = record["song_id"]
        metrics["boundary_indices"] = boundary_indices
        per_song.append(metrics)
        predictions.append(
            {
                "song_id": record["song_id"],
                "target": target_name,
                "method": candidate["method"],
                "boundary_indices": boundary_indices,
                "boundary_times": predicted,
                "target_times": target,
                "metrics": metrics,
            }
        )
    return {
        "target": target_name,
        "method": candidate["method"],
        "group": "03_official_msaf_modules",
        "candidate": candidate,
        "uses_allin1": False,
        "uses_manual_training": bool(candidate.get("uses_manual_training", False)),
        "aggregate": aggregate_metrics(per_song),
        "per_song": per_song,
        "predictions": predictions,
    }


def training_score_for_candidate(
    train_records: Sequence[Dict[str, Any]],
    candidate: Dict[str, Any],
    target_name: str,
    modules: Dict[str, Any],
) -> float:
    if len(train_records) < 2:
        return 0.0
    per_song = []
    for record in train_records:
        boundary_indices = run_msaf_candidate(record, candidate, modules)
        predicted = indices_to_times(record, boundary_indices)
        metrics = boundary_detection_metrics_seconds(target_for(record, target_name), predicted, float(record["tolerance_seconds"]))
        per_song.append(metrics)
    return float(aggregate_metrics(per_song)["f1"])


def evaluate_loso_tuned(
    records: Sequence[Dict[str, Any]],
    wrapper: Dict[str, Any],
    target_name: str,
    modules: Dict[str, Any],
) -> Dict[str, Any]:
    per_song = []
    predictions = []
    selected_rows = []
    for record in records:
        train_records = [item for item in records if item["song_id"] != record["song_id"]]
        scored = []
        for candidate in wrapper["grid"]:
            scored.append((training_score_for_candidate(train_records, candidate, target_name, modules), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        boundary_indices = run_msaf_candidate(record, best_candidate, modules)
        predicted = indices_to_times(record, boundary_indices)
        target = target_for(record, target_name)
        metrics = boundary_detection_metrics_seconds(target, predicted, float(record["tolerance_seconds"]))
        metrics["song_id"] = record["song_id"]
        metrics["boundary_indices"] = boundary_indices
        metrics["selected_training_f1"] = round(float(best_score), 6)
        metrics["selected_method"] = best_candidate["method"]
        per_song.append(metrics)
        selected_rows.append(
            {
                "song_id": record["song_id"],
                "target": target_name,
                "selected_training_f1": round(float(best_score), 6),
                "selected_candidate": best_candidate,
            }
        )
        predictions.append(
            {
                "song_id": record["song_id"],
                "target": target_name,
                "method": wrapper["method"],
                "selected_method": best_candidate["method"],
                "boundary_indices": boundary_indices,
                "boundary_times": predicted,
                "target_times": target,
                "metrics": metrics,
            }
        )
    return {
        "target": target_name,
        "method": wrapper["method"],
        "group": "03_official_msaf_modules",
        "candidate": {"method": wrapper["method"], "note": wrapper["note"]},
        "uses_allin1": False,
        "uses_manual_training": True,
        "aggregate": aggregate_metrics(per_song),
        "per_song": per_song,
        "selected_candidates": selected_rows,
        "predictions": predictions,
    }


def fixed_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "method": "official_msaf_foote_fixed_full_bar_features",
            "algorithm": "foote",
            "feature_set": "full",
            "config": {
                "M_gaussian": 8,
                "m_median": 3,
                "L_peaks": 8,
                "bound_norm_feats": "min_max",
            },
            "uses_manual_training": False,
        },
        {
            "method": "official_msaf_sf_fixed_full_bar_features",
            "algorithm": "sf",
            "feature_set": "full",
            "config": {
                "M_gaussian": 8,
                "m_embedded": 3,
                "k_nearest": 0.04,
                "Mp_adaptive": 8,
                "offset_thres": 0.05,
                "bound_norm_feats": np.inf,
            },
            "uses_manual_training": False,
        },
    ]


def foote_grid() -> List[Dict[str, Any]]:
    candidates = []
    for feature_set in ("mfcc", "timbre_chroma", "full"):
        for kernel in (4, 6, 8, 10, 12, 16):
            for median in (1, 3, 5):
                for peaks in (4, 6, 8, 12, 16):
                    candidates.append(
                        {
                            "method": f"official_msaf_foote_{feature_set}_M{kernel}_med{median}_L{peaks}",
                            "algorithm": "foote",
                            "feature_set": feature_set,
                            "config": {
                                "M_gaussian": kernel,
                                "m_median": median,
                                "L_peaks": peaks,
                                "bound_norm_feats": "min_max",
                            },
                            "uses_manual_training": True,
                        }
                    )
    return candidates


def sf_grid() -> List[Dict[str, Any]]:
    candidates = []
    for feature_set in ("timbre_chroma", "full"):
        for gaussian in (4, 6, 8, 10):
            for embedded in (2, 3):
                for adaptive in (6, 8, 12):
                    for offset in (0.03, 0.05, 0.08):
                        candidates.append(
                            {
                                "method": (
                                    f"official_msaf_sf_{feature_set}_M{gaussian}_emb{embedded}_"
                                    f"Mp{adaptive}_off{str(offset).replace('.', 'p')}"
                                ),
                                "algorithm": "sf",
                                "feature_set": feature_set,
                                "config": {
                                    "M_gaussian": gaussian,
                                    "m_embedded": embedded,
                                    "k_nearest": 0.04,
                                    "Mp_adaptive": adaptive,
                                    "offset_thres": offset,
                                    "bound_norm_feats": np.inf,
                                },
                                "uses_manual_training": True,
                            }
                        )
    return candidates


def tuned_wrappers(family: str) -> List[Dict[str, Any]]:
    wrappers = []
    if family in {"all", "foote"}:
        wrappers.append(
            {
                "method": "official_msaf_foote_bar_features_loso_tuned",
                "note": "Official MSAF Foote Segmenter, using existing bar-level features to avoid slow MSAF audio preprocessing.",
                "grid": foote_grid(),
            }
        )
    if family in {"all", "sf"}:
        wrappers.append(
            {
                "method": "official_msaf_sf_bar_features_loso_tuned",
                "note": "Official MSAF Structural Features Segmenter, using existing bar-level features to avoid slow MSAF audio preprocessing.",
                "grid": sf_grid(),
            }
        )
    return wrappers


def summary_lines(results: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "# Official MSAF Module Segmentation",
        "",
        "MSAF's full `msaf.process` path timed out on full MP3 files in this environment, and CNMF cannot import because cvxopt fails to load a DLL. This run uses official MSAF Foote/SF Segmenter modules on the existing bar-level features.",
        "",
        "| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(results, key=lambda item: (item["target"], -float(item["aggregate"]["f1"])))
    for item in ordered:
        metrics = item["aggregate"]
        lines.append(
            f"| {item['target']} | {item['method']} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['target_count']} | {metrics['predicted_count']} |"
        )
    return lines


def save_code_snapshot(out_dir: Path) -> None:
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        Path("scripts/official_msaf_segmentation.py"),
        Path("scripts/external_structure_segmentation.py"),
    ):
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)


def write_artifacts(out_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    write_json(out_dir / "official_msaf_metrics.json", {"results": results})
    (out_dir / "official_msaf_summary.md").write_text("\n".join(summary_lines(results)) + "\n", encoding="utf-8")
    for result in results:
        method_dir = out_dir / "methods" / str(result["method"]) / str(result["target"])
        method_dir.mkdir(parents=True, exist_ok=True)
        write_json(method_dir / "metrics.json", {key: value for key, value in result.items() if key != "predictions"})
        write_jsonl(method_dir / "predictions.jsonl", result["predictions"])
        if "selected_candidates" in result:
            write_jsonl(method_dir / "selected_candidates.jsonl", result["selected_candidates"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official MSAF algorithm modules on existing bar-level features.")
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/official_msaf_segmentation/round1"))
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--family", choices=["all", "foote", "sf"], default="all")
    args = parser.parse_args()

    records = load_song_records(args.bars_dir)
    if len(records) < 2:
        raise SystemExit("Need at least two songs with signal_bars outputs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_code_snapshot(args.out_dir)
    modules = import_msaf_modules()

    results: List[Dict[str, Any]] = []
    if not args.skip_fixed:
        for candidate in fixed_candidates():
            for target_name in TARGETS:
                print(f"Evaluating {target_name}: {candidate['method']}", flush=True)
                results.append(evaluate_fixed(records, candidate, target_name, modules))
                write_artifacts(args.out_dir, results)

    if not args.fixed_only:
        for wrapper in tuned_wrappers(args.family):
            for target_name in TARGETS:
                print(f"Evaluating {target_name}: {wrapper['method']}", flush=True)
                results.append(evaluate_loso_tuned(records, wrapper, target_name, modules))
                write_artifacts(args.out_dir, results)

    write_artifacts(args.out_dir, results)
    print(f"Wrote summary: {args.out_dir / 'official_msaf_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
