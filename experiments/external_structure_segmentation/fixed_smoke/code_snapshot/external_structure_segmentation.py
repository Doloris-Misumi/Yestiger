import argparse
import copy
import importlib.util
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import lil_matrix

from search_music_boundary_detectors import load_song_records, target_for
from callability_signal_experiment import boundary_detection_metrics_seconds, load_json, write_json


TARGETS = ("manual_fine", "manual_coarse")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def zscore_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def candidate_indices(record: Dict[str, Any]) -> List[int]:
    return [index for index, row in enumerate(record["rows"]) if float(row["start"]) > 0.0]


def index_to_time(record: Dict[str, Any], index: int) -> float:
    rows = record["rows"]
    index = int(np.clip(index, 1, len(rows) - 1))
    return float(rows[index]["start"])


def indices_to_times(record: Dict[str, Any], indices: Sequence[int]) -> List[float]:
    rows = record["rows"]
    output = []
    for index in sorted(set(int(item) for item in indices)):
        if 0 < index < len(rows):
            output.append(float(rows[index]["start"]))
    return output


def target_boundary_count_rate(records: Sequence[Dict[str, Any]], target_name: str) -> float:
    total_targets = 0
    total_candidates = 0
    for record in records:
        total_targets += len(target_for(record, target_name))
        total_candidates += len(candidate_indices(record))
    return total_targets / total_candidates if total_candidates else 0.0


def estimated_count(
    train_records: Sequence[Dict[str, Any]],
    test_record: Dict[str, Any],
    target_name: str,
    scale: float,
) -> int:
    rate = target_boundary_count_rate(train_records, target_name)
    count = int(round(rate * len(candidate_indices(test_record)) * scale))
    return int(np.clip(count, 0, max(0, len(test_record["rows"]) - 2)))


def bar_feature_matrix(record: Dict[str, Any], feature_set: str) -> np.ndarray:
    rows = record["rows"]
    matrix = []
    for row in rows:
        signal = row.get("signal_features") or {}
        novelty = row.get("novelty") or {}
        chroma = [float(value) for value in signal.get("chroma_mean", [])][:12]
        mfcc = [float(value) for value in signal.get("mfcc_mean", [])][:10]
        scalars = [
            float(signal.get("energy", 0.0)),
            float(signal.get("onset", 0.0)),
            float(signal.get("spectral_centroid", 0.0)),
            float(signal.get("spectral_bandwidth", 0.0)),
            float(signal.get("spectral_rolloff", 0.0)),
            float(signal.get("spectral_flatness", 0.0)),
            float(signal.get("vocal_density_proxy", 0.0)),
            float(signal.get("beat_stability", 0.0)),
            float(novelty.get("timbre", 0.0)),
            float(novelty.get("harmony", 0.0)),
            float(novelty.get("energy", 0.0)),
            float(novelty.get("onset", 0.0)),
            float(novelty.get("fused", 0.0)),
        ]
        if feature_set == "chroma":
            values = chroma
        elif feature_set == "mfcc":
            values = mfcc
        elif feature_set == "timbre_chroma":
            values = mfcc + chroma
        elif feature_set == "full":
            values = scalars + mfcc + chroma
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        matrix.append(values)
    if not matrix:
        return np.zeros((0, 1), dtype=np.float32)
    max_len = max(len(row) for row in matrix)
    padded = [row + [0.0] * (max_len - len(row)) for row in matrix]
    return zscore_columns(np.asarray(padded, dtype=np.float32))


def cosine_self_similarity(features: np.ndarray, smooth_sigma: float = 0.0) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    x = zscore_columns(features)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-6)
    sim = np.matmul(x, x.T)
    sim = np.clip((sim + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    np.fill_diagonal(sim, 1.0)
    if smooth_sigma > 0:
        sim = gaussian_filter(sim, sigma=float(smooth_sigma)).astype(np.float32)
        np.fill_diagonal(sim, 1.0)
    return sim


def foote_novelty_from_similarity(sim: np.ndarray, kernel_size: int = 8) -> np.ndarray:
    n = int(sim.shape[0])
    half = max(1, int(kernel_size))
    scores = np.zeros(n, dtype=np.float32)
    if n < 2 * half + 1:
        return scores
    kernel = np.zeros((2 * half, 2 * half), dtype=np.float32)
    kernel[:half, :half] = 1.0
    kernel[half:, half:] = 1.0
    kernel[:half, half:] = -1.0
    kernel[half:, :half] = -1.0
    kernel -= np.mean(kernel)
    denom = float(np.sum(np.abs(kernel))) or 1.0
    kernel /= denom
    for index in range(half, n - half):
        block = sim[index - half : index + half, index - half : index + half]
        scores[index] = float(np.sum(block * kernel))
    return minmax(scores)


def local_contrast_scores(sim: np.ndarray, window: int = 4) -> np.ndarray:
    n = int(sim.shape[0])
    scores = np.zeros(n, dtype=np.float32)
    width = max(1, int(window))
    for index in range(1, n):
        left = sim[max(0, index - width) : index, max(0, index - width) : index]
        right = sim[index : min(n, index + width), index : min(n, index + width)]
        cross = sim[max(0, index - width) : index, index : min(n, index + width)]
        if left.size == 0 or right.size == 0 or cross.size == 0:
            continue
        scores[index] = 0.5 * (float(np.mean(left)) + float(np.mean(right))) - float(np.mean(cross))
    return minmax(scores)


def suppress_and_select(
    scores: Sequence[float],
    count: int,
    min_gap: int,
    allowed_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    if count <= 0:
        return []
    allowed = set(int(item) for item in allowed_indices) if allowed_indices is not None else None
    ranked = []
    for index, score in enumerate(scores):
        if index <= 0:
            continue
        if allowed is not None and index not in allowed:
            continue
        ranked.append((index, float(score)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    selected: List[int] = []
    gap = max(1, int(min_gap))
    for index, _score in ranked:
        if all(abs(index - item) >= gap for item in selected):
            selected.append(index)
            if len(selected) >= count:
                break
    return sorted(selected)


def adjust_boundary_count(
    boundaries: Sequence[int],
    scores: Sequence[float],
    desired_count: int,
    min_gap: int,
    n_bars: int,
) -> List[int]:
    desired = int(np.clip(desired_count, 0, max(0, n_bars - 1)))
    current = sorted(set(int(item) for item in boundaries if 0 < int(item) < n_bars))
    if desired <= 0:
        return []
    if len(current) > desired:
        ranked = sorted(current, key=lambda index: float(scores[index]) if index < len(scores) else 0.0, reverse=True)
        return sorted(ranked[:desired])
    if len(current) == desired:
        return current
    allowed = [index for index in range(1, n_bars) if index not in current]
    extras = suppress_and_select(scores, desired - len(current), min_gap=min_gap, allowed_indices=allowed)
    merged = sorted(set(current + extras))
    if len(merged) > desired:
        ranked = sorted(merged, key=lambda index: float(scores[index]) if index < len(scores) else 0.0, reverse=True)
        return sorted(ranked[:desired])
    return merged


def cbm_length_penalty(length: int, period: int = 8) -> float:
    if length <= 0:
        return 1.0
    anchors = [4, 6, 8, 12, 16, 24, 32]
    anchors.extend([period * k for k in range(1, 8)])
    distance = min(abs(length - anchor) for anchor in anchors)
    return distance / max(1.0, float(period))


def cbm_block_dp_segments(
    sim: np.ndarray,
    min_size: int,
    max_size: int,
    penalty_weight: float,
    period: int = 8,
) -> List[Tuple[int, int]]:
    n = int(sim.shape[0])
    if n <= 1:
        return [(0, n)]
    min_len = max(1, int(min_size))
    max_len = max(min_len, int(max_size))
    scores = np.full(n + 1, -np.inf, dtype=np.float64)
    previous = np.zeros(n + 1, dtype=np.int32)
    scores[0] = 0.0
    prefix = np.pad(sim, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

    def block_sum(start: int, end: int) -> float:
        return float(prefix[end, end] - prefix[start, end] - prefix[end, start] + prefix[start, start])

    for end in range(1, n + 1):
        start_min = max(0, end - max_len)
        start_max = max(0, end - min_len)
        if end < min_len:
            start_candidates = [0]
        else:
            start_candidates = range(start_min, start_max + 1)
        for start in start_candidates:
            length = end - start
            if length <= 0 or length > max_len:
                continue
            total = block_sum(start, end)
            diag = float(length)
            denom = max(1.0, float(length * length - length))
            homogeneity = (total - diag) / denom
            length_cost = cbm_length_penalty(length, period=period)
            segment_score = homogeneity * length - penalty_weight * length_cost * max(4.0, math.sqrt(length))
            candidate = scores[start] + segment_score
            if candidate > scores[end]:
                scores[end] = candidate
                previous[end] = start

    segments = []
    cursor = n
    seen = set()
    while cursor > 0 and cursor not in seen:
        seen.add(cursor)
        start = int(previous[cursor])
        if start >= cursor:
            start = max(0, cursor - min_len)
        segments.append((start, cursor))
        cursor = start
    if cursor != 0:
        return [(0, n)]
    return list(reversed(segments))


def predict_foote(record: Dict[str, Any], train_records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> List[int]:
    features = bar_feature_matrix(record, str(candidate["feature_set"]))
    sim = cosine_self_similarity(features, smooth_sigma=float(candidate.get("smooth_sigma", 0.0)))
    novelty = foote_novelty_from_similarity(sim, kernel_size=int(candidate["kernel_size"]))
    count = estimated_count(train_records, record, target_name, float(candidate.get("scale", 1.0)))
    return suppress_and_select(novelty, count=count, min_gap=int(candidate.get("min_gap", 2)))


def predict_cbm(record: Dict[str, Any], train_records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> List[int]:
    features = bar_feature_matrix(record, str(candidate["feature_set"]))
    sim = cosine_self_similarity(features, smooth_sigma=float(candidate.get("smooth_sigma", 0.0)))
    segments = cbm_block_dp_segments(
        sim,
        min_size=int(candidate["min_size"]),
        max_size=int(candidate["max_size"]),
        penalty_weight=float(candidate["penalty_weight"]),
        period=int(candidate.get("period", 8)),
    )
    boundaries = [start for start, _end in segments[1:]]
    if bool(candidate.get("fill_to_loso_count", False)):
        novelty = 0.65 * foote_novelty_from_similarity(sim, kernel_size=int(candidate.get("kernel_size", 8)))
        novelty += 0.35 * local_contrast_scores(sim, window=max(2, int(candidate.get("kernel_size", 8)) // 2))
        desired = estimated_count(train_records, record, target_name, float(candidate.get("scale", 1.0)))
        boundaries = adjust_boundary_count(
            boundaries,
            scores=minmax(novelty),
            desired_count=desired,
            min_gap=int(candidate.get("min_gap", 2)),
            n_bars=len(record["rows"]),
        )
    return sorted(set(boundaries))


def predict_agglomerative(record: Dict[str, Any], train_records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> List[int]:
    from sklearn.cluster import AgglomerativeClustering

    features = bar_feature_matrix(record, str(candidate["feature_set"]))
    n = features.shape[0]
    if n <= 2:
        return []
    count = estimated_count(train_records, record, target_name, float(candidate.get("scale", 1.0)))
    n_segments = int(np.clip(count + 1, 2, max(2, n - 1)))
    connectivity = lil_matrix((n, n), dtype=np.int8)
    for index in range(n - 1):
        connectivity[index, index + 1] = 1
        connectivity[index + 1, index] = 1
    model = AgglomerativeClustering(
        n_clusters=n_segments,
        linkage=str(candidate.get("linkage", "ward")),
        connectivity=connectivity.tocsr(),
    )
    labels = model.fit_predict(features)
    boundaries = [index for index in range(1, n) if int(labels[index]) != int(labels[index - 1])]
    if len(boundaries) != count:
        sim = cosine_self_similarity(features, smooth_sigma=float(candidate.get("smooth_sigma", 0.0)))
        novelty = foote_novelty_from_similarity(sim, kernel_size=int(candidate.get("kernel_size", 8)))
        boundaries = adjust_boundary_count(boundaries, novelty, count, int(candidate.get("min_gap", 2)), n)
    return sorted(set(boundaries))


def predict_spectral(record: Dict[str, Any], train_records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> List[int]:
    from sklearn.cluster import SpectralClustering

    features = bar_feature_matrix(record, str(candidate["feature_set"]))
    n = features.shape[0]
    if n <= 2:
        return []
    count = estimated_count(train_records, record, target_name, float(candidate.get("scale", 1.0)))
    n_segments = int(np.clip(count + 1, 2, max(2, min(n - 1, 30))))
    sim = cosine_self_similarity(features, smooth_sigma=float(candidate.get("smooth_sigma", 0.0)))
    sim = np.maximum(sim, 1e-4)
    model = SpectralClustering(
        n_clusters=n_segments,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=23,
    )
    labels = model.fit_predict(sim)
    boundaries = [index for index in range(1, n) if int(labels[index]) != int(labels[index - 1])]
    novelty = foote_novelty_from_similarity(sim, kernel_size=int(candidate.get("kernel_size", 8)))
    boundaries = adjust_boundary_count(boundaries, novelty, count, int(candidate.get("min_gap", 2)), n)
    return sorted(set(boundaries))


def predict_fixed(record: Dict[str, Any], train_records: Sequence[Dict[str, Any]], candidate: Dict[str, Any], target_name: str) -> List[int]:
    kind = str(candidate["kind"])
    if kind == "cbm_dp":
        return predict_cbm(record, train_records, candidate, target_name)
    if kind == "foote":
        return predict_foote(record, train_records, candidate, target_name)
    if kind == "agglomerative":
        return predict_agglomerative(record, train_records, candidate, target_name)
    if kind == "spectral":
        return predict_spectral(record, train_records, candidate, target_name)
    raise ValueError(f"Unknown candidate kind: {kind}")


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
) -> Dict[str, Any]:
    per_song = []
    prediction_rows = []
    for record in records:
        train_records = [item for item in records if item["song_id"] != record["song_id"]]
        boundary_indices = predict_fixed(record, train_records, candidate, target_name)
        predicted = indices_to_times(record, boundary_indices)
        target = target_for(record, target_name)
        metrics = boundary_detection_metrics_seconds(target, predicted, float(record["tolerance_seconds"]))
        metrics["song_id"] = record["song_id"]
        metrics["boundary_indices"] = boundary_indices
        per_song.append(metrics)
        prediction_rows.append(
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
        "group": candidate["group"],
        "candidate": candidate,
        "uses_manual_training": bool(candidate.get("uses_manual_training", False)),
        "aggregate": aggregate_metrics(per_song),
        "per_song": per_song,
        "predictions": prediction_rows,
    }


def training_score_for_candidate(
    train_records: Sequence[Dict[str, Any]],
    candidate: Dict[str, Any],
    target_name: str,
) -> float:
    if len(train_records) < 2:
        return 0.0
    per_song = []
    for record in train_records:
        inner_train = [item for item in train_records if item["song_id"] != record["song_id"]]
        if not inner_train:
            inner_train = train_records
        boundary_indices = predict_fixed(record, inner_train, candidate, target_name)
        predicted = indices_to_times(record, boundary_indices)
        metrics = boundary_detection_metrics_seconds(target_for(record, target_name), predicted, float(record["tolerance_seconds"]))
        per_song.append(metrics)
    return float(aggregate_metrics(per_song)["f1"])


def evaluate_loso_tuned(
    records: Sequence[Dict[str, Any]],
    wrapper: Dict[str, Any],
    target_name: str,
) -> Dict[str, Any]:
    per_song = []
    prediction_rows = []
    selected_rows = []
    grid = wrapper["grid"]
    for record in records:
        train_records = [item for item in records if item["song_id"] != record["song_id"]]
        scored = []
        for candidate in grid:
            scored.append((training_score_for_candidate(train_records, candidate, target_name), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        boundary_indices = predict_fixed(record, train_records, best_candidate, target_name)
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
        prediction_rows.append(
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
        "group": wrapper["group"],
        "candidate": {key: value for key, value in wrapper.items() if key != "grid"},
        "uses_manual_training": True,
        "aggregate": aggregate_metrics(per_song),
        "per_song": per_song,
        "selected_candidates": selected_rows,
        "predictions": prediction_rows,
    }


def fixed_candidates() -> List[Dict[str, Any]]:
    return [
        {
            "method": "cbm_dp_chroma_fixed",
            "group": "01_cbm_block_dp",
            "kind": "cbm_dp",
            "feature_set": "chroma",
            "min_size": 4,
            "max_size": 16,
            "penalty_weight": 0.10,
            "smooth_sigma": 0.5,
            "uses_manual_training": False,
        },
        {
            "method": "cbm_dp_full_fixed",
            "group": "01_cbm_block_dp",
            "kind": "cbm_dp",
            "feature_set": "full",
            "min_size": 4,
            "max_size": 16,
            "penalty_weight": 0.08,
            "smooth_sigma": 0.5,
            "uses_manual_training": False,
        },
        {
            "method": "foote_checkerboard_full_loso_count",
            "group": "02_classic_msaf_like",
            "kind": "foote",
            "feature_set": "full",
            "kernel_size": 8,
            "smooth_sigma": 0.5,
            "scale": 1.0,
            "min_gap": 2,
            "uses_manual_training": True,
        },
        {
            "method": "agglomerative_full_loso_count",
            "group": "02_classic_msaf_like",
            "kind": "agglomerative",
            "feature_set": "full",
            "kernel_size": 8,
            "smooth_sigma": 0.5,
            "scale": 1.0,
            "min_gap": 2,
            "linkage": "ward",
            "uses_manual_training": True,
        },
        {
            "method": "spectral_full_loso_count",
            "group": "02_classic_msaf_like",
            "kind": "spectral",
            "feature_set": "full",
            "kernel_size": 8,
            "smooth_sigma": 0.5,
            "scale": 1.0,
            "min_gap": 2,
            "uses_manual_training": True,
        },
    ]


def cbm_grid(fill_to_count: bool) -> List[Dict[str, Any]]:
    candidates = []
    for feature_set in ("chroma", "timbre_chroma", "full"):
        for min_size in (3, 4, 6, 8):
            for max_size in (12, 16, 20, 24, 32):
                if max_size < min_size:
                    continue
                for penalty_weight in (0.0, 0.04, 0.08, 0.12, 0.20):
                    for smooth_sigma in (0.0, 0.5, 1.0):
                        candidate = {
                            "method": (
                                f"cbm_dp_{feature_set}_min{min_size}_max{max_size}_"
                                f"pen{str(penalty_weight).replace('.', 'p')}_smooth{str(smooth_sigma).replace('.', 'p')}"
                            ),
                            "group": "01_cbm_block_dp",
                            "kind": "cbm_dp",
                            "feature_set": feature_set,
                            "min_size": min_size,
                            "max_size": max_size,
                            "penalty_weight": penalty_weight,
                            "smooth_sigma": smooth_sigma,
                            "uses_manual_training": True,
                            "fill_to_loso_count": fill_to_count,
                            "kernel_size": 8,
                            "scale": 1.0,
                            "min_gap": 2,
                        }
                        if fill_to_count:
                            candidate["method"] += "_fill"
                            for scale in (0.75, 0.90, 1.00, 1.10, 1.25):
                                scaled = copy.deepcopy(candidate)
                                scaled["scale"] = scale
                                scaled["method"] += f"_scale{str(scale).replace('.', 'p')}"
                                candidates.append(scaled)
                        else:
                            candidates.append(candidate)
    return candidates


def foote_grid() -> List[Dict[str, Any]]:
    candidates = []
    for feature_set in ("chroma", "mfcc", "timbre_chroma", "full"):
        for kernel_size in (4, 6, 8, 10, 12, 16):
            for smooth_sigma in (0.0, 0.5, 1.0):
                for scale in (0.70, 0.85, 1.00, 1.15, 1.30):
                    for min_gap in (1, 2, 3, 4):
                        candidates.append(
                            {
                                "method": (
                                    f"foote_{feature_set}_k{kernel_size}_smooth{str(smooth_sigma).replace('.', 'p')}_"
                                    f"scale{str(scale).replace('.', 'p')}_gap{min_gap}"
                                ),
                                "group": "02_classic_msaf_like",
                                "kind": "foote",
                                "feature_set": feature_set,
                                "kernel_size": kernel_size,
                                "smooth_sigma": smooth_sigma,
                                "scale": scale,
                                "min_gap": min_gap,
                                "uses_manual_training": True,
                            }
                        )
    return candidates


def agglomerative_grid() -> List[Dict[str, Any]]:
    candidates = []
    for feature_set in ("mfcc", "timbre_chroma", "full"):
        for scale in (0.70, 0.85, 1.00, 1.15, 1.30):
            for min_gap in (1, 2, 3):
                candidates.append(
                    {
                        "method": f"agglomerative_{feature_set}_scale{str(scale).replace('.', 'p')}_gap{min_gap}",
                        "group": "02_classic_msaf_like",
                        "kind": "agglomerative",
                        "feature_set": feature_set,
                        "kernel_size": 8,
                        "smooth_sigma": 0.5,
                        "scale": scale,
                        "min_gap": min_gap,
                        "linkage": "ward",
                        "uses_manual_training": True,
                    }
                )
    return candidates


def spectral_grid() -> List[Dict[str, Any]]:
    candidates = []
    for feature_set in ("chroma", "timbre_chroma", "full"):
        for kernel_size in (6, 8, 10, 12):
            for smooth_sigma in (0.0, 0.5, 1.0):
                for scale in (0.70, 0.85, 1.00, 1.15):
                    candidates.append(
                        {
                            "method": (
                                f"spectral_{feature_set}_k{kernel_size}_smooth{str(smooth_sigma).replace('.', 'p')}_"
                                f"scale{str(scale).replace('.', 'p')}"
                            ),
                            "group": "02_classic_msaf_like",
                            "kind": "spectral",
                            "feature_set": feature_set,
                            "kernel_size": kernel_size,
                            "smooth_sigma": smooth_sigma,
                            "scale": scale,
                            "min_gap": 2,
                            "uses_manual_training": True,
                        }
                    )
    return candidates


def tuned_wrappers(args: argparse.Namespace) -> List[Dict[str, Any]]:
    wrappers = [
        {
            "method": "cbm_dp_loso_tuned",
            "group": "01_cbm_block_dp",
            "grid": cbm_grid(fill_to_count=False),
        },
        {
            "method": "cbm_dp_plus_novelty_fill_loso_tuned",
            "group": "01_cbm_block_dp",
            "grid": cbm_grid(fill_to_count=True),
        },
        {
            "method": "foote_checkerboard_loso_tuned",
            "group": "02_classic_msaf_like",
            "grid": foote_grid(),
        },
        {
            "method": "agglomerative_loso_tuned",
            "group": "02_classic_msaf_like",
            "grid": agglomerative_grid(),
        },
    ]
    if args.include_spectral:
        wrappers.append(
            {
                "method": "spectral_loso_tuned",
                "group": "02_classic_msaf_like",
                "grid": spectral_grid(),
            }
        )
    return wrappers


def summary_lines(results: Sequence[Dict[str, Any]], baseline_metrics: Optional[Dict[str, Any]]) -> List[str]:
    lines = [
        "# External Music Structure Segmentation",
        "",
        "This experiment compares non-allin1 music-structure boundary detectors on the same bar grid used by the signal-callability experiments.",
        "",
        "## Baselines From Previous Run",
        "",
        "| Target | Method | Uses allin1 | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    if baseline_metrics:
        comparison = baseline_metrics.get("music_boundary_comparison") or {}
        for target_name in TARGETS:
            for method_name in ("allin1_structure", "fused_novelty_topk"):
                metrics = (comparison.get(target_name) or {}).get(method_name)
                if not metrics:
                    continue
                uses_allin1 = "true" if method_name == "allin1_structure" else "false"
                lines.append(
                    f"| {target_name} | {method_name} | {uses_allin1} | "
                    f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | "
                    f"{metrics['target_count']} | {metrics['predicted_count']} |"
                )
    lines.extend(
        [
            "",
            "## New Methods",
            "",
            "| Group | Target | Method | Uses manual training | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    ordered = sorted(results, key=lambda item: (item["target"], item["group"], -float(item["aggregate"]["f1"])))
    for item in ordered:
        metrics = item["aggregate"]
        lines.append(
            f"| {item['group']} | {item['target']} | {item['method']} | "
            f"{str(bool(item['uses_manual_training'])).lower()} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | "
            f"{metrics['target_count']} | {metrics['predicted_count']} |"
        )
    lines.extend(["", "## Best Per Target", ""])
    for target_name in TARGETS:
        subset = [item for item in results if item["target"] == target_name]
        if not subset:
            continue
        best = max(subset, key=lambda item: float(item["aggregate"]["f1"]))
        best_pure_fixed = max(
            [item for item in subset if not item["uses_manual_training"]],
            key=lambda item: float(item["aggregate"]["f1"]),
            default=None,
        )
        lines.append(
            f"- `{target_name}` best new method: `{best['method']}` "
            f"({best['group']}), F1={best['aggregate']['f1']:.3f}."
        )
        if best_pure_fixed:
            lines.append(
                f"- `{target_name}` best fixed unsupervised method: `{best_pure_fixed['method']}`, "
                f"F1={best_pure_fixed['aggregate']['f1']:.3f}."
            )
    return lines


def method_notes_lines(package_report: Dict[str, Any]) -> List[str]:
    return [
        "# Method Notes",
        "",
        "## 01_cbm_block_dp",
        "",
        "- Inspired by CBM / autosimilarity segmentation: build a bar-level self-similarity matrix, score candidate blocks by within-block homogeneity, and solve the best whole-song segmentation by dynamic programming.",
        "- `cbm_dp_*_fixed` uses fixed hyperparameters and no manual labels.",
        "- `cbm_dp_loso_tuned` selects CBM hyperparameters on the other songs only.",
        "- `cbm_dp_plus_novelty_fill_loso_tuned` first runs CBM-DP, then adjusts the number of boundaries with a Foote-style novelty curve; this is a CBM + local novelty hybrid.",
        "",
        "## 02_classic_msaf_like",
        "",
        "- `foote_checkerboard_*` is a checkerboard novelty detector on the self-similarity matrix.",
        "- `agglomerative_*` is a temporally constrained agglomerative clustering baseline, similar in spirit to classic MSAF segmentation recipes.",
        "- `spectral_*` is an optional spectral-clustering segmentation baseline on the self-similarity matrix.",
        "",
        "## 03_modern_embedding_interface",
        "",
        "- MERT/BEATs/OpenBEATs are stronger modern feature extractors, but they are not direct structure segmenters. The intended extension is: extract bar-level embeddings, replace `bar_feature_matrix`, then reuse CBM-DP/Foote/spectral decoding.",
        "- This run does not download large pretrained models; it keeps the interface and method note separate from the measured results.",
        "",
        "## External Package Check",
        "",
        "```json",
        json.dumps(package_report, ensure_ascii=False, indent=2),
        "```",
    ]


def package_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    for package_name in ("barmuscomp", "as_seg", "msaf", "transformers"):
        spec = importlib.util.find_spec(package_name)
        report[package_name] = {"installed": spec is not None, "origin": spec.origin if spec else None}
    if report.get("as_seg", {}).get("installed"):
        try:
            import as_seg  # noqa: F401

            report["as_seg"]["importable"] = True
            report["as_seg"]["note"] = "Imported successfully."
        except SyntaxError as exc:
            report["as_seg"]["importable"] = False
            report["as_seg"]["note"] = f"SyntaxError on Python {'.'.join(map(str, [3, 9]))}: {exc}"
        except Exception as exc:  # pragma: no cover - diagnostic only
            report["as_seg"]["importable"] = False
            report["as_seg"]["note"] = f"{type(exc).__name__}: {exc}"
    return report


def save_code_snapshot(out_dir: Path) -> None:
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        Path("scripts/external_structure_segmentation.py"),
        Path("scripts/search_music_boundary_detectors.py"),
        Path("scripts/callability_signal_experiment.py"),
    ):
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)


def write_result_artifacts(out_dir: Path, results: Sequence[Dict[str, Any]], baseline_metrics: Optional[Dict[str, Any]]) -> None:
    write_json(out_dir / "external_structure_metrics.json", {"results": results})
    summary_path = out_dir / "external_structure_summary.md"
    summary_path.write_text("\n".join(summary_lines(results, baseline_metrics)) + "\n", encoding="utf-8")
    notes_path = out_dir / "METHOD_NOTES.md"
    notes_path.write_text("\n".join(method_notes_lines(package_report())) + "\n", encoding="utf-8")

    for result in results:
        safe_method = str(result["method"]).replace("/", "_")
        method_dir = out_dir / "methods" / str(result["group"]) / safe_method / str(result["target"])
        method_dir.mkdir(parents=True, exist_ok=True)
        write_json(method_dir / "metrics.json", {key: value for key, value in result.items() if key != "predictions"})
        write_jsonl(method_dir / "predictions.jsonl", result["predictions"])
        if "selected_candidates" in result:
            write_jsonl(method_dir / "selected_candidates.jsonl", result["selected_candidates"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate non-allin1 music structure segmentation methods.")
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/external_structure_segmentation/round1"))
    parser.add_argument("--baseline-metrics", type=Path, default=Path("experiments/signal_callability/aggregate_signal_metrics.json"))
    parser.add_argument("--include-spectral", action="store_true", help="Also run spectral clustering tuning. Slower and noisier.")
    parser.add_argument("--fixed-only", action="store_true", help="Only run fixed smoke-test candidates.")
    args = parser.parse_args()

    records = load_song_records(args.bars_dir)
    if len(records) < 2:
        raise SystemExit("Need at least two songs with signal_bars outputs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_code_snapshot(args.out_dir)

    baseline_metrics = load_json(args.baseline_metrics) if args.baseline_metrics.exists() else None
    results: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = fixed_candidates()
    for candidate in candidates:
        for target_name in TARGETS:
            print(f"Evaluating {target_name}: {candidate['method']}", flush=True)
            results.append(evaluate_fixed(records, candidate, target_name))
            write_result_artifacts(args.out_dir, results, baseline_metrics)

    if not args.fixed_only:
        for wrapper in tuned_wrappers(args):
            for target_name in TARGETS:
                print(f"Evaluating {target_name}: {wrapper['method']}", flush=True)
                results.append(evaluate_loso_tuned(records, wrapper, target_name))
                write_result_artifacts(args.out_dir, results, baseline_metrics)

    write_result_artifacts(args.out_dir, results, baseline_metrics)
    print(f"Wrote summary: {args.out_dir / 'external_structure_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
