import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from callability_signal_experiment import (  # noqa: E402
    annotation_music_segments,
    boundary_detection_metrics_seconds,
    interior_boundaries_from_segments,
    load_json,
    struct_music_segments,
    write_json,
)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_song_records(bars_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for bars_path in sorted(bars_dir.glob("*/*.signal_bars.jsonl")):
        song_id = bars_path.parent.name
        metrics_path = bars_path.parent / f"{song_id}.signal_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        annotation_path = Path(str(metrics["annotation"]))
        struct_path = Path(str(metrics["struct"]))
        rows = list(load_jsonl(bars_path))
        rows.sort(key=lambda item: int(item["bar_index"]))
        full_bar_durations = [
            float(row["features"]["duration"])
            for row in rows
            if row.get("bar_kind") == "full_bar"
        ]
        tolerance_seconds = float(np.median(full_bar_durations)) if full_bar_durations else 2.5
        annotation = load_json(annotation_path)
        struct = load_json(struct_path)
        records.append(
            {
                "song_id": song_id,
                "rows": rows,
                "annotation": annotation,
                "struct": struct,
                "tolerance_seconds": tolerance_seconds,
                "fine_target": interior_boundaries_from_segments(annotation_music_segments(annotation, coarse=False)),
                "coarse_target": interior_boundaries_from_segments(annotation_music_segments(annotation, coarse=True)),
                "allin1": interior_boundaries_from_segments(struct_music_segments(struct)),
            }
        )
    return records


def candidate_rows(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = record["rows"]
    output = []
    for index, row in enumerate(rows):
        time = float(row["start"])
        if time <= 0.0:
            continue
        output.append({"index": index, "time": time, "row": row})
    return output


def target_for(record: Dict[str, Any], target_name: str) -> List[float]:
    return list(record["fine_target"] if target_name == "manual_fine" else record["coarse_target"])


def boundary_rate(records: Sequence[Dict[str, Any]], held_out: str, target_name: str) -> float:
    target_count = 0
    candidate_count = 0
    for record in records:
        if record["song_id"] == held_out:
            continue
        target_count += len(target_for(record, target_name))
        candidate_count += len(candidate_rows(record))
    return target_count / candidate_count if candidate_count else 0.0


def allin1_ratio(records: Sequence[Dict[str, Any]], held_out: str, target_name: str) -> float:
    target_count = 0
    allin1_count = 0
    for record in records:
        if record["song_id"] == held_out:
            continue
        target_count += len(target_for(record, target_name))
        allin1_count += len(record["allin1"])
    return target_count / allin1_count if allin1_count else 1.0


def row_signal_score(row: Dict[str, Any], mode: str) -> float:
    features = row.get("features") or {}
    signal = row.get("signal_features") or {}
    novelty = row.get("novelty") or {}
    struct = float(features.get("allin1_struct_boundary", 0.0))
    overlap = float(features.get("allin1_struct_label_overlap", 0.0))
    fused = float(novelty.get("fused", 0.0))
    timbre = float(novelty.get("timbre", 0.0))
    harmony = float(novelty.get("harmony", 0.0))
    energy_novelty = float(novelty.get("energy", 0.0))
    onset_novelty = float(novelty.get("onset", 0.0))
    energy = float(signal.get("energy", 0.0))
    onset = float(signal.get("onset", 0.0))
    vocal = float(signal.get("vocal_density_proxy", 0.0))

    audio = 0.34 * fused + 0.18 * timbre + 0.16 * harmony + 0.16 * energy_novelty + 0.16 * onset_novelty
    rhythmic = 0.26 * fused + 0.24 * onset_novelty + 0.18 * energy_novelty + 0.16 * onset + 0.16 * energy
    contrast = 0.30 * fused + 0.20 * timbre + 0.20 * harmony + 0.15 * abs(energy - vocal) + 0.15 * onset_novelty
    if mode == "audio":
        return audio
    if mode == "rhythmic":
        return rhythmic
    if mode == "contrast":
        return contrast
    if mode == "hybrid_audio":
        return 0.48 * struct + 0.08 * overlap + 0.44 * audio
    if mode == "hybrid_rhythmic":
        return 0.50 * struct + 0.08 * overlap + 0.42 * rhythmic
    if mode == "hybrid_contrast":
        return 0.50 * struct + 0.08 * overlap + 0.42 * contrast
    raise ValueError(f"Unknown score mode: {mode}")


def feature_vector(record: Dict[str, Any], candidate: Dict[str, Any], include_struct: bool) -> List[float]:
    rows = record["rows"]
    index = int(candidate["index"])
    row = candidate["row"]
    features = row.get("features") or {}
    signal = row.get("signal_features") or {}
    novelty = row.get("novelty") or {}
    previous = rows[index - 1] if index > 0 else row
    previous_signal = previous.get("signal_features") or {}
    previous_novelty = previous.get("novelty") or {}
    n = max(1, len(rows) - 1)

    values = [
        index / n,
        float(signal.get("energy", 0.0)),
        float(signal.get("onset", 0.0)),
        float(signal.get("vocal_density_proxy", 0.0)),
        float(signal.get("beat_stability", 0.0)),
        float(signal.get("spectral_centroid", 0.0)),
        float(signal.get("spectral_bandwidth", 0.0)),
        float(signal.get("spectral_flatness", 0.0)),
        float(novelty.get("fused", 0.0)),
        float(novelty.get("timbre", 0.0)),
        float(novelty.get("harmony", 0.0)),
        float(novelty.get("energy", 0.0)),
        float(novelty.get("onset", 0.0)),
        float(signal.get("energy", 0.0)) - float(previous_signal.get("energy", 0.0)),
        float(signal.get("onset", 0.0)) - float(previous_signal.get("onset", 0.0)),
        float(signal.get("vocal_density_proxy", 0.0)) - float(previous_signal.get("vocal_density_proxy", 0.0)),
        float(novelty.get("fused", 0.0)) - float(previous_novelty.get("fused", 0.0)),
        row_signal_score(row, "audio"),
        row_signal_score(row, "rhythmic"),
        row_signal_score(row, "contrast"),
    ]
    mfcc = [float(value) for value in signal.get("mfcc_mean", [])]
    chroma = [float(value) for value in signal.get("chroma_mean", [])]
    values.extend(mfcc[:6])
    values.extend(chroma)
    if include_struct:
        values.extend(
            [
                float(features.get("allin1_struct_boundary", 0.0)),
                float(features.get("allin1_struct_label_overlap", 0.0)),
                row_signal_score(row, "hybrid_audio"),
                row_signal_score(row, "hybrid_rhythmic"),
                row_signal_score(row, "hybrid_contrast"),
            ]
        )
    return values


def target_labels_for_candidates(record: Dict[str, Any], target_name: str) -> List[int]:
    target = target_for(record, target_name)
    tolerance = float(record["tolerance_seconds"])
    labels = []
    for candidate in candidate_rows(record):
        time = float(candidate["time"])
        labels.append(1 if any(abs(time - item) <= tolerance for item in target) else 0)
    return labels


def make_model(kind: str) -> Any:
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for music-boundary model search.") from exc

    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.7, class_weight="balanced", max_iter=3000, random_state=23),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=220,
            max_depth=7,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=23,
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=90,
            learning_rate=0.045,
            max_depth=2,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=23,
        )
    raise ValueError(f"Unknown model kind: {kind}")


def learned_scores(
    records: Sequence[Dict[str, Any]],
    held_out: str,
    test_record: Dict[str, Any],
    target_name: str,
    model_kind: str,
    include_struct: bool,
) -> List[Tuple[float, float]]:
    x_train = []
    y_train = []
    for record in records:
        if record["song_id"] == held_out:
            continue
        candidates = candidate_rows(record)
        labels = target_labels_for_candidates(record, target_name)
        for candidate, label in zip(candidates, labels):
            x_train.append(feature_vector(record, candidate, include_struct=include_struct))
            y_train.append(label)
    candidates = candidate_rows(test_record)
    if not candidates or len(set(y_train)) < 2:
        return []
    model = make_model(model_kind)
    model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.int32))
    x_test = np.asarray([feature_vector(test_record, candidate, include_struct=include_struct) for candidate in candidates], dtype=np.float32)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_test), dtype=np.float32)[:, 1]
    else:
        scores = np.asarray(model.decision_function(x_test), dtype=np.float32)
    return [(float(candidate["time"]), float(score)) for candidate, score in zip(candidates, scores)]


def select_topk(scored_times: Sequence[Tuple[float, float]], count: int) -> List[float]:
    if count <= 0:
        return []
    selected = sorted(scored_times, key=lambda item: item[1], reverse=True)[: min(count, len(scored_times))]
    return sorted(time for time, _ in selected)


def away_from_existing(time: float, existing: Sequence[float], tolerance: float) -> bool:
    return all(abs(time - item) > tolerance for item in existing)


def predict_boundaries(record: Dict[str, Any], records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> List[float]:
    song_id = str(record["song_id"])
    scale = float(candidate.get("scale", 1.0))
    candidates = candidate_rows(record)
    estimated_count = int(round(boundary_rate(records, song_id, target_name) * len(candidates) * scale))
    allin1_count_estimate = int(round(allin1_ratio(records, song_id, target_name) * len(record["allin1"]) * scale))

    kind = str(candidate["kind"])
    if kind == "allin1":
        return list(record["allin1"])
    if kind == "formula_topk":
        scored = [(float(item["time"]), row_signal_score(item["row"], str(candidate["score_mode"]))) for item in candidates]
        return select_topk(scored, estimated_count)
    if kind == "learned_topk":
        scored = learned_scores(
            records,
            song_id,
            record,
            target_name,
            model_kind=str(candidate["model"]),
            include_struct=bool(candidate["include_struct"]),
        )
        return select_topk(scored, estimated_count)
    if kind == "hybrid_add_formula":
        base = list(record["allin1"])
        extra_count = max(0, allin1_count_estimate - len(base))
        tolerance = 0.5 * float(record["tolerance_seconds"])
        scored = [
            (float(item["time"]), row_signal_score(item["row"], str(candidate["score_mode"])))
            for item in candidates
            if away_from_existing(float(item["time"]), base, tolerance)
        ]
        return sorted(base + select_topk(scored, extra_count))
    if kind == "hybrid_add_learned":
        base = list(record["allin1"])
        extra_count = max(0, allin1_count_estimate - len(base))
        tolerance = 0.5 * float(record["tolerance_seconds"])
        scored = [
            (time, score)
            for time, score in learned_scores(
                records,
                song_id,
                record,
                target_name,
                model_kind=str(candidate["model"]),
                include_struct=bool(candidate["include_struct"]),
            )
            if away_from_existing(time, base, tolerance)
        ]
        return sorted(base + select_topk(scored, extra_count))
    raise ValueError(f"Unknown candidate kind: {kind}")


def aggregate_metrics(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(item["tp"]) for item in items)
    fp = sum(int(item["fp"]) for item in items)
    fn = sum(int(item["fn"]) for item in items)
    target_count = sum(int(item["target_count"]) for item in items)
    predicted_count = sum(int(item["predicted_count"]) for item in items)
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
        "mean_tolerance_seconds": round(float(np.mean([float(item["tolerance_seconds"]) for item in items])), 6) if items else 0.0,
    }


def candidate_grid(family: str = "all") -> List[Dict[str, Any]]:
    candidates = [{"method": "allin1_structure", "kind": "allin1", "uses_allin1": True}]
    if family == "allin1_plus":
        base_families = {"allin1_plus"}
    elif family == "audio_only":
        base_families = {"audio_formula", "audio_learned"}
    elif family == "learned_topk":
        base_families = {"audio_learned", "hybrid_learned"}
    else:
        base_families = {"audio_formula", "audio_learned", "hybrid_learned", "allin1_plus"}

    for score_mode in ("audio", "rhythmic", "contrast"):
        if "audio_formula" in base_families:
            for scale in (0.8, 1.0, 1.2):
                candidates.append(
                    {
                        "method": f"audio_formula_{score_mode}_scale{str(scale).replace('.', 'p')}",
                        "kind": "formula_topk",
                        "score_mode": score_mode,
                        "scale": scale,
                        "uses_allin1": False,
                    }
                )
    for model in ("logreg", "rf", "gb"):
        for include_struct in (False, True):
            prefix = "hybrid_learned" if include_struct else "audio_learned"
            if prefix not in base_families:
                continue
            for scale in (0.8, 1.0, 1.15):
                candidates.append(
                    {
                        "method": f"{prefix}_{model}_scale{str(scale).replace('.', 'p')}",
                        "kind": "learned_topk",
                        "model": model,
                        "include_struct": include_struct,
                        "scale": scale,
                        "uses_allin1": include_struct,
                    }
                )
    if "allin1_plus" in base_families:
        for score_mode in ("audio", "rhythmic", "contrast"):
            for scale in (0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8):
                candidates.append(
                    {
                        "method": f"allin1_plus_{score_mode}_scale{str(scale).replace('.', 'p')}",
                        "kind": "hybrid_add_formula",
                        "score_mode": score_mode,
                        "scale": scale,
                        "uses_allin1": True,
                    }
                )
        for model in ("logreg", "rf", "gb"):
            for scale in (0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8):
                candidates.append(
                    {
                        "method": f"allin1_plus_learned_{model}_scale{str(scale).replace('.', 'p')}",
                        "kind": "hybrid_add_learned",
                        "model": model,
                        "include_struct": True,
                        "scale": scale,
                        "uses_allin1": True,
                    }
                )
    return candidates


def evaluate_candidate(records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> Dict[str, Any]:
    per_song = []
    for record in records:
        target = target_for(record, target_name)
        predicted = predict_boundaries(record, records, candidate, target_name)
        metrics = boundary_detection_metrics_seconds(target, predicted, float(record["tolerance_seconds"]))
        metrics["song_id"] = record["song_id"]
        per_song.append(metrics)
    aggregate = aggregate_metrics(per_song)
    return {
        "target": target_name,
        "method": candidate["method"],
        "candidate": candidate,
        "uses_allin1": bool(candidate.get("uses_allin1", False)),
        "aggregate": aggregate,
        "per_song": per_song,
    }


def summary_lines(results: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "# Music Boundary Detector Search",
        "",
        "| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        results,
        key=lambda item: (item["target"], -float(item["aggregate"]["f1"])),
    )
    for item in ordered:
        aggregate = item["aggregate"]
        lines.append(
            f"| {item['target']} | {item['method']} | {str(item['uses_allin1']).lower()} | "
            f"{aggregate['precision']:.3f} | {aggregate['recall']:.3f} | {aggregate['f1']:.3f} | "
            f"{aggregate['target_count']} | {aggregate['predicted_count']} |"
        )
    for target_name in ("manual_fine", "manual_coarse"):
        subset = [item for item in results if item["target"] == target_name]
        if not subset:
            continue
        best = max(subset, key=lambda item: float(item["aggregate"]["f1"]))
        pure_subset = [item for item in subset if not item["uses_allin1"]]
        best_pure = max(pure_subset, key=lambda item: float(item["aggregate"]["f1"])) if pure_subset else None
        lines.extend(["", f"Best `{target_name}`: `{best['method']}` F1={best['aggregate']['f1']:.3f}."])
        if best_pure:
            lines.append(f"Best pure-audio `{target_name}`: `{best_pure['method']}` F1={best_pure['aggregate']['f1']:.3f}.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Search music-section boundary detectors against manual fine/coarse targets.")
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/signal_callability_music_boundary_search/round1"))
    parser.add_argument("--family", default="all", choices=["all", "audio_only", "learned_topk", "allin1_plus"])
    args = parser.parse_args()

    records = load_song_records(args.bars_dir)
    if len(records) < 2:
        raise SystemExit("Need at least two songs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for candidate in candidate_grid(args.family):
        for target_name in ("manual_fine", "manual_coarse"):
            print(f"Searching {target_name}: {candidate['method']}", flush=True)
            result = evaluate_candidate(records, candidate, target_name)
            results.append(result)
            write_json(args.out_dir / "music_boundary_search_metrics.json", {"results": results})
            (args.out_dir / "music_boundary_search_summary.md").write_text(
                "\n".join(summary_lines(results)) + "\n",
                encoding="utf-8",
            )
    write_json(args.out_dir / "music_boundary_search_metrics.json", {"results": results})
    (args.out_dir / "music_boundary_search_summary.md").write_text(
        "\n".join(summary_lines(results)) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote music boundary search summary: {args.out_dir / 'music_boundary_search_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
