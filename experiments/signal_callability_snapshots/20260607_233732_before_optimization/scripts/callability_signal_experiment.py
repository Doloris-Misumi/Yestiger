import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import librosa
import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_bar_level_dataset import as_float, best_segment, build_samples, rounded
from build_pipeline_dataset import resolve_struct_path


ROLE_VOCAB = ["keepspace", "rhythmcall", "mix", "underground_gei"]
ROLE_COLORS = {
    "keepspace": "#8a8f98",
    "rhythmcall": "#2a9d8f",
    "mix": "#e76f51",
    "underground_gei": "#7b2cbf",
}
STRUCT_MIX_CONTEXTS = {
    "intro",
    "inst",
    "instrumental",
    "instrumental_break",
    "solo",
    "bridge",
    "pre_chorus",
    "pre_chorus_build",
    "post_chorus",
}
STRUCT_CHORUS_CONTEXTS = {"chorus", "post_chorus", "outro"}
COARSE_MUSIC_LABELS = ["start", "intro", "verse", "chorus", "inst", "solo", "bridge", "outro", "end"]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def zscore_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def frame_index(time: float, sr: int, hop_length: int, num_frames: int) -> int:
    return int(np.clip(round(time * sr / hop_length), 0, max(0, num_frames - 1)))


def frame_slice(start: float, end: float, sr: int, hop_length: int, num_frames: int) -> slice:
    left = frame_index(start, sr, hop_length, num_frames)
    right = frame_index(end, sr, hop_length, num_frames)
    if right <= left:
        right = min(num_frames, left + 1)
    return slice(left, right)


def safe_mean(values: np.ndarray, span: slice) -> float:
    chunk = values[span]
    if chunk.size == 0:
        return 0.0
    return float(np.nanmean(chunk))


def safe_vector_mean(values: np.ndarray, span: slice) -> List[float]:
    chunk = values[:, span]
    if chunk.size == 0:
        return [0.0 for _ in range(values.shape[0])]
    return np.nanmean(chunk, axis=1).astype(float).tolist()


def normalize_role(role: Any) -> str:
    text = str(role or "keepspace")
    return text if text in ROLE_VOCAB else "keepspace"


def resolve_annotation_paths(args: argparse.Namespace) -> List[Path]:
    annotations_dir = args.annotations_dir
    if args.annotation:
        return [args.annotation]
    if args.all:
        return sorted(annotations_dir.glob("*/*.annotation.json"))
    if args.song:
        direct = Path(args.song)
        if direct.exists():
            return [direct]
        candidate = annotations_dir / args.song / f"{args.song}.annotation.json"
        if candidate.exists():
            return [candidate]
    raise SystemExit("Provide --song, --annotation, or --all.")


def resolve_audio_path(annotation: Dict[str, Any], struct: Dict[str, Any], explicit_audio: Optional[Path], root: Path) -> Path:
    if explicit_audio:
        return explicit_audio if explicit_audio.is_absolute() else root / explicit_audio
    song = annotation.get("song") or {}
    raw_audio = song.get("audio_path") or struct.get("path")
    if not raw_audio:
        raise SystemExit("No audio path found in annotation or struct.")
    audio_path = Path(str(raw_audio))
    return audio_path if audio_path.is_absolute() else root / audio_path


def trim_feature_lengths(features: Dict[str, Any]) -> Dict[str, Any]:
    lengths = [
        len(features["rms"]),
        len(features["onset"]),
        len(features["centroid"]),
        len(features["bandwidth"]),
        len(features["rolloff"]),
        len(features["flatness"]),
        features["mfcc"].shape[1],
        features["chroma"].shape[1],
    ]
    n = min(lengths)
    result = dict(features)
    for key in ("rms", "onset", "centroid", "bandwidth", "rolloff", "flatness"):
        result[key] = np.asarray(result[key][:n], dtype=np.float32)
    result["mfcc"] = np.asarray(result["mfcc"][:, :n], dtype=np.float32)
    result["chroma"] = np.asarray(result["chroma"][:, :n], dtype=np.float32)
    return result


def compute_frame_features(audio_path: Path, sr: int, hop_length: int) -> Dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    stft = librosa.stft(y, n_fft=2048, hop_length=hop_length)
    magnitude = np.abs(stft)
    power = magnitude**2
    power_db = librosa.power_to_db(power, ref=np.max)

    rms = librosa.feature.rms(S=magnitude)[0]
    onset = librosa.onset.onset_strength(S=power_db, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    mel = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=40)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), n_mfcc=13)
    chroma = librosa.feature.chroma_stft(S=power, sr=sr)

    features = trim_feature_lengths(
        {
            "rms": rms,
            "onset": onset,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "flatness": flatness,
            "mfcc": mfcc,
            "chroma": chroma,
        }
    )

    rms_norm = minmax(gaussian_filter1d(features["rms"], sigma=4.0))
    onset_norm = minmax(gaussian_filter1d(features["onset"], sigma=4.0))
    flatness_norm = minmax(gaussian_filter1d(features["flatness"], sigma=4.0))
    centroid_norm = minmax(gaussian_filter1d(features["centroid"], sigma=4.0))
    midrange_balance = np.clip(1.0 - np.abs(centroid_norm - 0.45) / 0.45, 0.0, 1.0)

    # This is a proxy, not a vocal separator: vocally dense passages tend to
    # be energetic, less percussive, less noise-like, and concentrated around
    # a mid-frequency spectral centroid range. It is intentionally lightweight
    # for fast project-scale experiments.
    vocal_density_proxy = minmax(
        (0.35 * rms_norm + 0.25 * (1.0 - onset_norm) + 0.25 * (1.0 - flatness_norm) + 0.15 * midrange_balance)
        * rms_norm
    )

    features.update(
        {
            "sr": sr,
            "hop_length": hop_length,
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "rms_norm": rms_norm,
            "onset_norm": onset_norm,
            "midrange_balance": midrange_balance,
            "vocal_density_proxy": vocal_density_proxy,
        }
    )
    return features


def add_targets(rows: List[Dict[str, Any]], annotation: Dict[str, Any]) -> None:
    call_spans = [span for span in annotation.get("call_spans") or [] if isinstance(span, dict)]
    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        role, span, overlap = best_segment(call_spans, start, end, "call_role")
        if not span or overlap < 0.1:
            role = "keepspace"
            overlap = 0.0
        row.setdefault("target", {})
        row["target"]["call_role"] = normalize_role(role)
        row["target"]["call_overlap"] = round(float(overlap), 6)


def add_signal_features(rows: List[Dict[str, Any]], frame_features: Dict[str, Any]) -> None:
    sr = int(frame_features["sr"])
    hop = int(frame_features["hop_length"])
    n = len(frame_features["rms"])
    full_bar_durations = [float(row["features"]["duration"]) for row in rows if row.get("bar_kind") == "full_bar"]
    median_bar = float(np.median(full_bar_durations)) if full_bar_durations else 2.5

    timbre_vectors = []
    harmony_vectors = []
    energy_values = []
    onset_values = []

    for row in rows:
        span = frame_slice(float(row["start"]), float(row["end"]), sr, hop, n)
        signal = {
            "energy": safe_mean(frame_features["rms_norm"], span),
            "onset": safe_mean(frame_features["onset_norm"], span),
            "spectral_centroid": safe_mean(minmax(frame_features["centroid"]), span),
            "spectral_bandwidth": safe_mean(minmax(frame_features["bandwidth"]), span),
            "spectral_rolloff": safe_mean(minmax(frame_features["rolloff"]), span),
            "spectral_flatness": safe_mean(minmax(frame_features["flatness"]), span),
            "midrange_balance": safe_mean(frame_features["midrange_balance"], span),
            "vocal_density_proxy": safe_mean(frame_features["vocal_density_proxy"], span),
            "mfcc_mean": safe_vector_mean(frame_features["mfcc"], span),
            "chroma_mean": safe_vector_mean(frame_features["chroma"], span),
        }
        duration_ratio = float(row["features"].get("bar_duration_ratio") or 1.0)
        bar_regularity = math.exp(-3.0 * abs(duration_ratio - 1.0))
        sources = row.get("grid_sources") or []
        downbeat_quality = 1.0 if "struct_downbeat" in sources else 0.65 if "extrapolated_downbeat" in sources else 0.35
        signal["beat_stability"] = clamp(0.65 * bar_regularity + 0.35 * downbeat_quality)
        row["signal_features"] = {key: rounded(value) if isinstance(value, float) else value for key, value in signal.items()}

        timbre_vectors.append(
            [
                signal["spectral_centroid"],
                signal["spectral_bandwidth"],
                signal["spectral_rolloff"],
                signal["spectral_flatness"],
                *signal["mfcc_mean"],
            ]
        )
        harmony_vectors.append(signal["chroma_mean"])
        energy_values.append(signal["energy"])
        onset_values.append(signal["onset"])

    timbre = zscore_columns(np.asarray(timbre_vectors, dtype=np.float32))
    harmony = zscore_columns(np.asarray(harmony_vectors, dtype=np.float32))
    energy = zscore_columns(np.asarray(energy_values, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    onset = zscore_columns(np.asarray(onset_values, dtype=np.float32).reshape(-1, 1)).reshape(-1)

    timbre_novelty = np.concatenate([[0.0], np.linalg.norm(np.diff(timbre, axis=0), axis=1) / math.sqrt(timbre.shape[1])])
    harmony_novelty = np.concatenate([[0.0], np.linalg.norm(np.diff(harmony, axis=0), axis=1) / math.sqrt(harmony.shape[1])])
    energy_novelty = np.concatenate([[0.0], np.abs(np.diff(energy))])
    onset_novelty = np.concatenate([[0.0], np.abs(np.diff(onset))])

    timbre_novelty = minmax(gaussian_filter1d(timbre_novelty, sigma=1.0))
    harmony_novelty = minmax(gaussian_filter1d(harmony_novelty, sigma=1.0))
    energy_novelty = minmax(gaussian_filter1d(energy_novelty, sigma=1.0))
    onset_novelty = minmax(gaussian_filter1d(onset_novelty, sigma=1.0))
    fused_novelty = minmax(
        0.30 * timbre_novelty + 0.25 * harmony_novelty + 0.25 * energy_novelty + 0.20 * onset_novelty
    )

    for index, row in enumerate(rows):
        row["novelty"] = {
            "timbre": rounded(float(timbre_novelty[index])),
            "harmony": rounded(float(harmony_novelty[index])),
            "energy": rounded(float(energy_novelty[index])),
            "onset": rounded(float(onset_novelty[index])),
            "fused": rounded(float(fused_novelty[index])),
        }

    similarity_features = zscore_columns(
        np.hstack(
            [
                np.asarray(timbre_vectors, dtype=np.float32),
                np.asarray(harmony_vectors, dtype=np.float32),
                np.asarray(energy_values, dtype=np.float32).reshape(-1, 1),
                np.asarray(onset_values, dtype=np.float32).reshape(-1, 1),
            ]
        )
    )
    norms = np.linalg.norm(similarity_features, axis=1, keepdims=True)
    normalized = similarity_features / np.maximum(norms, 1e-6)
    self_similarity = np.clip((normalized @ normalized.T + 1.0) / 2.0, 0.0, 1.0)
    return self_similarity


def compute_callability(row: Dict[str, Any]) -> Dict[str, float]:
    signal = row["signal_features"]
    novelty = row["novelty"]["fused"]
    struct_label = str((row.get("features") or {}).get("allin1_struct_label") or "unknown")
    instrumental_prior = 1.0 if struct_label in STRUCT_MIX_CONTEXTS else 0.0
    chorus_prior = 1.0 if struct_label in STRUCT_CHORUS_CONTEXTS else 0.0

    energy = float(signal["energy"])
    onset = float(signal["onset"])
    beat = float(signal["beat_stability"])
    vocal = float(signal["vocal_density_proxy"])

    mix = 0.30 * energy + 0.22 * beat + 0.23 * novelty + 0.15 * (1.0 - vocal) + 0.10 * instrumental_prior
    rhythm = 0.28 * beat + 0.22 * onset + 0.20 * energy + 0.15 * vocal + 0.15 * chorus_prior
    keepspace = 0.38 * vocal + 0.22 * (1.0 - energy) + 0.18 * (1.0 - onset) + 0.12 * (1.0 - beat) + 0.10 * (1.0 - novelty)
    gei = 0.35 * energy + 0.25 * onset + 0.15 * beat + 0.15 * (1.0 - vocal) + 0.10 * chorus_prior
    return {
        "keepspace": rounded(clamp(keepspace)),
        "rhythmcall": rounded(clamp(rhythm)),
        "mix": rounded(clamp(mix)),
        "underground_gei": rounded(clamp(gei)),
    }


def predict_callability_rule(row: Dict[str, Any]) -> str:
    scores = row["callability"]
    signal = row["signal_features"]
    novelty = float(row["novelty"]["fused"])
    struct_label = str((row.get("features") or {}).get("allin1_struct_label") or "unknown")
    energy = float(signal["energy"])
    onset = float(signal["onset"])
    vocal = float(signal["vocal_density_proxy"])
    instrumental_prior = struct_label in STRUCT_MIX_CONTEXTS

    if scores["keepspace"] >= max(scores["rhythmcall"], scores["mix"], scores["underground_gei"]) and scores["keepspace"] > 0.58:
        return "keepspace"
    if scores["underground_gei"] > 0.72 and energy > 0.65 and onset > 0.45 and vocal < 0.62:
        return "underground_gei"
    if scores["mix"] > 0.56 and vocal < 0.70 and (novelty > 0.30 or instrumental_prior):
        return "mix"
    if scores["rhythmcall"] > 0.40:
        return "rhythmcall"
    return max(scores.items(), key=lambda item: item[1])[0]


def predict_structure_baseline(row: Dict[str, Any]) -> str:
    label = str((row.get("features") or {}).get("allin1_struct_label") or "unknown")
    if label in {"end", "start", "unknown"}:
        return "keepspace"
    if label in STRUCT_MIX_CONTEXTS:
        return "mix"
    if label in {"verse", "chorus", "outro"}:
        return "rhythmcall"
    return "keepspace"


def add_predictions(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        row["callability"] = compute_callability(row)
        row["predictions"] = {
            "structure_baseline": predict_structure_baseline(row),
            "callability_rule": predict_callability_rule(row),
        }


def classification_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for target, pred in zip(y_true, y_pred) if target == pred)
    confusion = {target: {pred: 0 for pred in ROLE_VOCAB} for target in ROLE_VOCAB}
    for target, pred in zip(y_true, y_pred):
        if target in confusion and pred in confusion[target]:
            confusion[target][pred] += 1

    per_role = {}
    f1_values = []
    for role in ROLE_VOCAB:
        tp = confusion[role][role]
        fp = sum(confusion[other][role] for other in ROLE_VOCAB if other != role)
        fn = sum(confusion[role][other] for other in ROLE_VOCAB if other != role)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = sum(confusion[role].values())
        per_role[role] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": support,
        }
        if support:
            f1_values.append(f1)

    return {
        "accuracy": round(correct / total, 6) if total else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 6) if f1_values else 0.0,
        "total": total,
        "per_role": per_role,
        "confusion": confusion,
    }


def role_distribution(values: Sequence[str]) -> Dict[str, int]:
    counts = Counter(values)
    return {role: int(counts.get(role, 0)) for role in ROLE_VOCAB}


def merge_role_sequence(rows: Sequence[Dict[str, Any]], roles: Sequence[str], method: str) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    if not rows:
        return spans

    current_role = normalize_role(roles[0])
    current_start = float(rows[0]["start"])
    current_end = float(rows[0]["end"])
    current_indices = [int(rows[0]["bar_index"])]

    for row, role in zip(rows[1:], roles[1:]):
        role = normalize_role(role)
        start = float(row["start"])
        end = float(row["end"])
        if role == current_role and abs(start - current_end) <= 0.08:
            current_end = end
            current_indices.append(int(row["bar_index"]))
            continue
        spans.append(
            {
                "start": rounded(current_start),
                "end": rounded(current_end),
                "call_role": current_role,
                "bar_start": current_indices[0],
                "bar_end": current_indices[-1],
                "bars": len(current_indices),
                "method": method,
            }
        )
        current_role = role
        current_start = start
        current_end = end
        current_indices = [int(row["bar_index"])]

    spans.append(
        {
            "start": rounded(current_start),
            "end": rounded(current_end),
            "call_role": current_role,
            "bar_start": current_indices[0],
            "bar_end": current_indices[-1],
            "bars": len(current_indices),
            "method": method,
        }
    )
    return spans


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def role_at(spans: Sequence[Dict[str, Any]], time: float) -> str:
    for span in spans:
        if float(span["start"]) <= time < float(span["end"]):
            return normalize_role(span.get("call_role"))
    return "keepspace"


def span_change_times(spans: Sequence[Dict[str, Any]]) -> List[float]:
    return [float(span["start"]) for span in spans[1:]]


def boundary_detection_metrics_seconds(
    target: Sequence[float],
    predicted: Sequence[float],
    tolerance: float,
) -> Dict[str, Any]:
    target_items = [float(value) for value in target]
    predicted_items = [float(value) for value in predicted]
    matched_targets = set()
    tp = 0
    for pred in predicted_items:
        available = [
            index
            for index, target_time in enumerate(target_items)
            if index not in matched_targets and abs(target_time - pred) <= tolerance
        ]
        if not available:
            continue
        best = min(available, key=lambda index: abs(target_items[index] - pred))
        matched_targets.add(best)
        tp += 1
    fp = len(predicted_items) - tp
    fn = len(target_items) - tp
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
        "target_count": len(target_items),
        "predicted_count": len(predicted_items),
        "tolerance_seconds": round(float(tolerance), 6),
    }


def coarse_music_label(label: Any) -> str:
    text = str(label or "unknown")
    if text in {"start"}:
        return "start"
    if text in {"intro"}:
        return "intro"
    if text in {"verse", "pre_chorus", "pre_chorus_build", "chant"}:
        return "verse"
    if text in {"chorus", "post_chorus"}:
        return "chorus"
    if text in {"inst", "instrumental", "instrumental_break", "interlude"}:
        return "inst"
    if text in {"solo"}:
        return "solo"
    if text in {"bridge"}:
        return "bridge"
    if text in {"outro"}:
        return "outro"
    if text in {"end"}:
        return "end"
    return "unknown"


def annotation_music_segments(annotation: Dict[str, Any], coarse: bool = False) -> List[Dict[str, Any]]:
    raw_segments = [item for item in annotation.get("segments") or [] if isinstance(item, dict)]
    segments = []
    for item in raw_segments:
        start = as_float(item.get("start"))
        end = as_float(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        label = str(item.get("music_label") or "unknown")
        if coarse:
            label = coarse_music_label(label)
        segment = {"start": rounded(start), "end": rounded(end), "label": label}
        if coarse and segments and segments[-1]["label"] == label and abs(float(segments[-1]["end"]) - start) <= 0.08:
            segments[-1]["end"] = rounded(end)
        else:
            segments.append(segment)
    return segments


def struct_music_segments(struct: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = []
    for item in struct.get("segments") or []:
        if not isinstance(item, dict):
            continue
        start = as_float(item.get("start"))
        end = as_float(item.get("end"))
        if start is None or end is None or end <= start:
            continue
        label = coarse_music_label(item.get("label"))
        segment = {"start": rounded(start), "end": rounded(end), "label": label}
        if segments and segments[-1]["label"] == label and abs(float(segments[-1]["end"]) - start) <= 0.08:
            segments[-1]["end"] = rounded(end)
        else:
            segments.append(segment)
    return segments


def interior_boundaries_from_segments(segments: Sequence[Dict[str, Any]]) -> List[float]:
    if len(segments) <= 1:
        return []
    return [float(segment["start"]) for segment in segments[1:]]


def novelty_boundary_times(rows: Sequence[Dict[str, Any]], count: int) -> List[float]:
    if count <= 0:
        return []
    candidates = [
        (float(row["start"]), float((row.get("novelty") or {}).get("fused") or 0.0))
        for row in rows
        if float(row["start"]) > 0.0
    ]
    selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:count]
    return sorted(time for time, _ in selected)


def music_boundary_comparison(
    rows: Sequence[Dict[str, Any]],
    annotation: Dict[str, Any],
    struct: Dict[str, Any],
    tolerance_seconds: float,
) -> Dict[str, Any]:
    fine_target = interior_boundaries_from_segments(annotation_music_segments(annotation, coarse=False))
    coarse_target = interior_boundaries_from_segments(annotation_music_segments(annotation, coarse=True))
    struct_pred = interior_boundaries_from_segments(struct_music_segments(struct))
    return {
        "target_counts": {
            "manual_fine": len(fine_target),
            "manual_coarse": len(coarse_target),
            "allin1_struct": len(struct_pred),
        },
        "manual_fine": {
            "allin1_structure": boundary_detection_metrics_seconds(fine_target, struct_pred, tolerance_seconds),
            "fused_novelty_topk": boundary_detection_metrics_seconds(
                fine_target,
                novelty_boundary_times(rows, count=len(fine_target)),
                tolerance_seconds,
            ),
        },
        "manual_coarse": {
            "allin1_structure": boundary_detection_metrics_seconds(coarse_target, struct_pred, tolerance_seconds),
            "fused_novelty_topk": boundary_detection_metrics_seconds(
                coarse_target,
                novelty_boundary_times(rows, count=len(coarse_target)),
                tolerance_seconds,
            ),
        },
        "note": (
            "manual_fine uses all human music-section boundaries; manual_coarse collapses "
            "pre/post/refinement labels to the coarse allin1-like label set before comparison."
        ),
    }


def merged_span_metrics(
    target_spans: Sequence[Dict[str, Any]],
    predicted_spans: Sequence[Dict[str, Any]],
    tolerance_seconds: float,
) -> Dict[str, Any]:
    points = sorted(
        {
            rounded(float(span["start"]))
            for span in target_spans
        }
        | {rounded(float(span["end"])) for span in target_spans}
        | {rounded(float(span["start"])) for span in predicted_spans}
        | {rounded(float(span["end"])) for span in predicted_spans}
    )
    total = 0.0
    correct = 0.0
    intersection = {role: 0.0 for role in ROLE_VOCAB}
    union = {role: 0.0 for role in ROLE_VOCAB}
    confusion_duration = {target: {pred: 0.0 for pred in ROLE_VOCAB} for target in ROLE_VOCAB}

    for left, right in zip(points[:-1], points[1:]):
        if right <= left:
            continue
        mid = (left + right) / 2.0
        duration = right - left
        target_role = role_at(target_spans, mid)
        predicted_role = role_at(predicted_spans, mid)
        total += duration
        confusion_duration[target_role][predicted_role] += duration
        if target_role == predicted_role:
            correct += duration
            intersection[target_role] += duration
        for role in ROLE_VOCAB:
            if target_role == role or predicted_role == role:
                union[role] += duration

    role_iou = {
        role: round(intersection[role] / union[role], 6) if union[role] else 0.0
        for role in ROLE_VOCAB
    }
    nonzero_ious = [value for role, value in role_iou.items() if union[role] > 0]
    return {
        "time_weighted_role_accuracy": round(correct / total, 6) if total else 0.0,
        "macro_role_iou": round(float(np.mean(nonzero_ious)), 6) if nonzero_ious else 0.0,
        "role_iou": role_iou,
        "boundary": boundary_detection_metrics_seconds(
            span_change_times(target_spans),
            span_change_times(predicted_spans),
            tolerance=tolerance_seconds,
        ),
        "target_span_count": len(target_spans),
        "predicted_span_count": len(predicted_spans),
        "duration_seconds": round(total, 6),
        "confusion_duration": {
            target: {pred: round(value, 3) for pred, value in pred_items.items()}
            for target, pred_items in confusion_duration.items()
        },
    }


def write_merged_callbook(path: Path, song_id: str, method: str, spans: Sequence[Dict[str, Any]]) -> None:
    lines = [
        f"# Merged Signal Callbook: {song_id}",
        "",
        f"Method: `{method}`",
        "",
        "| Time | Role | Bars |",
        "|---:|---|---:|",
    ]
    for span in spans:
        lines.append(
            f"| {fmt_time(float(span['start']))}-{fmt_time(float(span['end']))} | "
            f"{span['call_role']} | {span['bars']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_merged_outputs(
    song_dir: Path,
    song_id: str,
    method: str,
    target_spans: Sequence[Dict[str, Any]],
    predicted_spans: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> None:
    write_json(
        song_dir / f"{song_id}.merged.{method}.call_spans.json",
        {
            "song_id": song_id,
            "method": method,
            "call_spans": list(predicted_spans),
            "comparison_to_manual_grid": metrics,
        },
    )
    write_merged_callbook(
        song_dir / f"{song_id}.merged.{method}.callbook.md",
        song_id,
        method,
        predicted_spans,
    )
    if method == "target_manual_grid":
        write_json(
            song_dir / f"{song_id}.merged.target_manual_grid.call_spans.json",
            {"song_id": song_id, "method": method, "call_spans": list(target_spans)},
        )


def write_target_merged_outputs(song_dir: Path, song_id: str, target_spans: Sequence[Dict[str, Any]]) -> None:
    write_json(
        song_dir / f"{song_id}.merged.target_manual_grid.call_spans.json",
        {"song_id": song_id, "method": "target_manual_grid", "call_spans": list(target_spans)},
    )
    write_merged_callbook(
        song_dir / f"{song_id}.merged.target_manual_grid.callbook.md",
        song_id,
        "target_manual_grid",
        target_spans,
    )


def call_role_boundaries(rows: Sequence[Dict[str, Any]]) -> List[int]:
    boundaries = []
    previous = None
    for row in rows:
        role = row["target"]["call_role"]
        index = int(row["bar_index"])
        if previous is not None and role != previous:
            boundaries.append(index)
        previous = role
    return boundaries


def struct_boundary_candidates(rows: Sequence[Dict[str, Any]]) -> List[int]:
    return [
        int(row["bar_index"])
        for row in rows
        if int(row["bar_index"]) > 0 and int((row.get("features") or {}).get("allin1_struct_boundary") or 0) == 1
    ]


def novelty_boundary_candidates(rows: Sequence[Dict[str, Any]], count: int) -> List[int]:
    if count <= 0:
        return []
    candidates = [
        (int(row["bar_index"]), float((row.get("novelty") or {}).get("fused") or 0.0))
        for row in rows
        if int(row["bar_index"]) > 0
    ]
    selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:count]
    return sorted(index for index, _ in selected)


def boundary_detection_metrics(target: Sequence[int], predicted: Sequence[int], tolerance: int = 1) -> Dict[str, Any]:
    target_set = set(int(value) for value in target)
    predicted_items = [int(value) for value in predicted]
    matched_targets = set()
    tp = 0
    for pred in predicted_items:
        available = [target for target in target_set if target not in matched_targets and abs(target - pred) <= tolerance]
        if not available:
            continue
        best = min(available, key=lambda target: abs(target - pred))
        matched_targets.add(best)
        tp += 1
    fp = len(predicted_items) - tp
    fn = len(target_set) - tp
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
        "target_count": len(target_set),
        "predicted_count": len(predicted_items),
        "tolerance_bars": tolerance,
    }


def aggregate_boundary_detection(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for method in ("structure_boundary", "fused_novelty_topk"):
        tp = fp = fn = 0
        target_count = predicted_count = 0
        for result in results:
            item = result["metrics"]["boundary_detection"][method]
            tp += int(item["tp"])
            fp += int(item["fp"])
            fn += int(item["fn"])
            target_count += int(item["target_count"])
            predicted_count += int(item["predicted_count"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        aggregate[method] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "target_count": target_count,
            "predicted_count": predicted_count,
            "tolerance_bars": 1,
        }
    return aggregate


def aggregate_music_boundary_comparison(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate: Dict[str, Any] = {}
    for target_name in ("manual_fine", "manual_coarse"):
        aggregate[target_name] = {}
        for method in ("allin1_structure", "fused_novelty_topk"):
            tp = fp = fn = 0
            target_count = predicted_count = 0
            tolerance_values = []
            for result in results:
                comparison = result["metrics"].get("music_boundary_comparison") or {}
                item = (comparison.get(target_name) or {}).get(method) or {}
                tp += int(item.get("tp", 0))
                fp += int(item.get("fp", 0))
                fn += int(item.get("fn", 0))
                target_count += int(item.get("target_count", 0))
                predicted_count += int(item.get("predicted_count", 0))
                tolerance_values.append(float(item.get("tolerance_seconds", 0.0)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            aggregate[target_name][method] = {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "target_count": target_count,
                "predicted_count": predicted_count,
                "mean_tolerance_seconds": round(float(np.mean(tolerance_values)), 6) if tolerance_values else 0.0,
            }
    aggregate["note"] = (
        "manual_fine compares against all human music segment boundaries. "
        "manual_coarse first folds fine labels into an allin1-like coarse label set, "
        "which is a fairer boundary-only comparison for allin1."
    )
    return aggregate


def aggregate_merged_span_comparison(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = {}
    for method in ("structure_baseline", "callability_rule", "loso_structure_rf", "loso_audio_rf"):
        confusion = {target: {pred: 0.0 for pred in ROLE_VOCAB} for target in ROLE_VOCAB}
        boundary_tp = boundary_fp = boundary_fn = 0
        boundary_target_count = boundary_predicted_count = 0
        tolerance_values = []
        found = False
        for result in results:
            comparison = (result["metrics"].get("merged_span_comparison") or {}).get(method)
            if not comparison:
                continue
            found = True
            for target in ROLE_VOCAB:
                for pred in ROLE_VOCAB:
                    confusion[target][pred] += float((comparison.get("confusion_duration") or {}).get(target, {}).get(pred, 0.0))
            boundary = comparison.get("boundary") or {}
            boundary_tp += int(boundary.get("tp", 0))
            boundary_fp += int(boundary.get("fp", 0))
            boundary_fn += int(boundary.get("fn", 0))
            boundary_target_count += int(boundary.get("target_count", 0))
            boundary_predicted_count += int(boundary.get("predicted_count", 0))
            tolerance_values.append(float(boundary.get("tolerance_seconds", 0.0)))
        if not found:
            continue

        total = sum(confusion[target][pred] for target in ROLE_VOCAB for pred in ROLE_VOCAB)
        correct = sum(confusion[role][role] for role in ROLE_VOCAB)
        role_iou = {}
        for role in ROLE_VOCAB:
            intersection = confusion[role][role]
            target_total = sum(confusion[role][pred] for pred in ROLE_VOCAB)
            pred_total = sum(confusion[target][role] for target in ROLE_VOCAB)
            union = target_total + pred_total - intersection
            role_iou[role] = round(intersection / union, 6) if union else 0.0
        nonzero_ious = []
        for role in ROLE_VOCAB:
            target_total = sum(confusion[role][pred] for pred in ROLE_VOCAB)
            pred_total = sum(confusion[target][role] for target in ROLE_VOCAB)
            if target_total + pred_total > 0:
                nonzero_ious.append(role_iou[role])

        boundary_precision = boundary_tp / (boundary_tp + boundary_fp) if boundary_tp + boundary_fp else 0.0
        boundary_recall = boundary_tp / (boundary_tp + boundary_fn) if boundary_tp + boundary_fn else 0.0
        boundary_f1 = (
            2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
            if boundary_precision + boundary_recall
            else 0.0
        )
        aggregate[method] = {
            "time_weighted_role_accuracy": round(correct / total, 6) if total else 0.0,
            "macro_role_iou": round(float(np.mean(nonzero_ious)), 6) if nonzero_ious else 0.0,
            "role_iou": role_iou,
            "boundary": {
                "precision": round(boundary_precision, 6),
                "recall": round(boundary_recall, 6),
                "f1": round(boundary_f1, 6),
                "tp": boundary_tp,
                "fp": boundary_fp,
                "fn": boundary_fn,
                "target_count": boundary_target_count,
                "predicted_count": boundary_predicted_count,
                "mean_tolerance_seconds": round(float(np.mean(tolerance_values)), 6) if tolerance_values else 0.0,
            },
            "duration_seconds": round(total, 6),
            "confusion_duration": {
                target: {pred: round(value, 3) for pred, value in pred_items.items()}
                for target, pred_items in confusion.items()
            },
        }
    return aggregate


def plot_callability(rows: List[Dict[str, Any]], path: Path, title: str) -> None:
    times = np.asarray([(float(row["start"]) + float(row["end"])) / 2.0 for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    for role, color in ROLE_COLORS.items():
        active = False
        span_start = None
        for row in rows:
            target = row["target"]["call_role"]
            if target == role and not active:
                active = True
                span_start = float(row["start"])
            if active and target != role:
                axes[0].axvspan(span_start, float(row["start"]), color=color, alpha=0.08, linewidth=0)
                axes[1].axvspan(span_start, float(row["start"]), color=color, alpha=0.08, linewidth=0)
                active = False
        if active:
            axes[0].axvspan(span_start, float(rows[-1]["end"]), color=color, alpha=0.08, linewidth=0)
            axes[1].axvspan(span_start, float(rows[-1]["end"]), color=color, alpha=0.08, linewidth=0)

    axes[0].plot(times, [row["signal_features"]["energy"] for row in rows], label="energy", color="#1f77b4")
    axes[0].plot(times, [row["signal_features"]["onset"] for row in rows], label="onset", color="#ff7f0e")
    axes[0].plot(times, [row["signal_features"]["vocal_density_proxy"] for row in rows], label="vocal density proxy", color="#2ca02c")
    axes[0].plot(times, [row["novelty"]["fused"] for row in rows], label="fused novelty", color="#d62728")
    axes[0].set_ylabel("Signal")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)
    axes[0].grid(alpha=0.2)

    for role, color in ROLE_COLORS.items():
        axes[1].plot(times, [row["callability"][role] for row in rows], label=role, color=color)
    axes[1].set_ylabel("Callability")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right", ncol=4, fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_self_similarity(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_title(title)
    ax.set_xlabel("Bar index")
    ax.set_ylabel("Bar index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confusion(confusion: Dict[str, Dict[str, int]], path: Path, title: str) -> None:
    matrix = np.asarray([[confusion[target][pred] for pred in ROLE_VOCAB] for target in ROLE_VOCAB], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(ROLE_VOCAB)), ROLE_VOCAB, rotation=30, ha="right")
    ax.set_yticks(range(len(ROLE_VOCAB)), ROLE_VOCAB)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_song_summary(path: Path, song_id: str, metrics: Dict[str, Any]) -> None:
    lines = [
        f"# Signal Callability Experiment: {song_id}",
        "",
        "## Metrics",
        "",
        "| Method | Accuracy | Macro-F1 | Bars |",
        "|---|---:|---:|---:|",
    ]
    for method in ("structure_baseline", "callability_rule"):
        item = metrics[method]
        lines.append(f"| {method} | {item['accuracy']:.3f} | {item['macro_f1']:.3f} | {item['total']} |")
    lines.extend(["", "## Target Role Distribution", ""])
    for role, count in metrics["target_distribution"].items():
        lines.append(f"- `{role}`: {count}")
    lines.extend(["", "## Boundary Detection", "", "| Method | Precision | Recall | F1 |", "|---|---:|---:|---:|"])
    for method in ("structure_boundary", "fused_novelty_topk"):
        item = metrics["boundary_detection"][method]
        lines.append(f"| {method} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |")
    if "music_boundary_comparison" in metrics:
        lines.extend(
            [
                "",
                "## Music Segment Boundary Comparison",
                "",
                "Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.",
                "",
                "| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        comparison = metrics["music_boundary_comparison"]
        for target in ("manual_fine", "manual_coarse"):
            for method in ("allin1_structure", "fused_novelty_topk"):
                item = comparison[target][method]
                lines.append(
                    f"| {target} | {method} | {item['precision']:.3f} | {item['recall']:.3f} | "
                    f"{item['f1']:.3f} | {item['target_count']} | {item['predicted_count']} |"
                )
    if "merged_span_comparison" in metrics:
        lines.extend(
            [
                "",
                "## Merged Span Comparison",
                "",
                "| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in ("structure_baseline", "callability_rule", "loso_structure_rf", "loso_audio_rf"):
            item = metrics["merged_span_comparison"].get(method)
            if not item:
                continue
            lines.append(
                f"| {method} | {item['time_weighted_role_accuracy']:.3f} | "
                f"{item['macro_role_iou']:.3f} | {item['boundary']['f1']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_song(annotation_path: Path, args: argparse.Namespace, root: Path) -> Dict[str, Any]:
    annotation = load_json(annotation_path)
    struct_path = args.struct or resolve_struct_path(annotation, annotation_path, args.struct_dir)
    if not struct_path:
        raise SystemExit(f"No matching struct found for {annotation_path}")
    struct = load_json(struct_path)
    audio_path = resolve_audio_path(annotation, struct, args.audio, root)
    song = annotation.get("song") or {}
    song_id = str(song.get("song_id") or annotation_path.parent.name)
    title = str(song.get("title") or song_id)

    rows, _ = build_samples(annotation, struct)
    add_targets(rows, annotation)
    frame_features = compute_frame_features(audio_path, sr=args.sr, hop_length=args.hop_length)
    self_similarity = add_signal_features(rows, frame_features)
    add_predictions(rows)

    y_true = [row["target"]["call_role"] for row in rows]
    structure_pred = [row["predictions"]["structure_baseline"] for row in rows]
    callability_pred = [row["predictions"]["callability_rule"] for row in rows]
    full_bar_durations = [float(row["features"]["duration"]) for row in rows if row.get("bar_kind") == "full_bar"]
    boundary_tolerance_seconds = float(np.median(full_bar_durations)) if full_bar_durations else 2.5
    target_merged = merge_role_sequence(rows, y_true, method="target_manual_grid")
    structure_merged = merge_role_sequence(rows, structure_pred, method="structure_baseline")
    callability_merged = merge_role_sequence(rows, callability_pred, method="callability_rule")
    merged_span_comparison = {
        "structure_baseline": merged_span_metrics(target_merged, structure_merged, boundary_tolerance_seconds),
        "callability_rule": merged_span_metrics(target_merged, callability_merged, boundary_tolerance_seconds),
    }
    metrics = {
        "song_id": song_id,
        "title": title,
        "annotation": str(annotation_path),
        "struct": str(struct_path),
        "audio": str(audio_path),
        "bars": len(rows),
        "target_distribution": role_distribution(y_true),
        "merged_span_comparison": merged_span_comparison,
        "feature_definition": {
            "frame_features": [
                "rms_energy",
                "onset_strength",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "spectral_flatness",
                "mfcc_13",
                "chroma_stft",
            ],
            "bar_features": [
                "energy",
                "onset",
                "vocal_density_proxy",
                "beat_stability",
                "timbre_novelty",
                "harmony_novelty",
                "energy_novelty",
                "onset_novelty",
                "fused_novelty",
            ],
            "note": "vocal_density_proxy is a spectral proxy, not source-separated vocals.",
        },
        "structure_baseline": classification_metrics(y_true, structure_pred),
        "callability_rule": classification_metrics(y_true, callability_pred),
    }
    target_boundaries = call_role_boundaries(rows)
    metrics["boundary_detection"] = {
        "target_boundaries": target_boundaries,
        "structure_boundary": boundary_detection_metrics(target_boundaries, struct_boundary_candidates(rows), tolerance=1),
        "fused_novelty_topk": boundary_detection_metrics(
            target_boundaries,
            novelty_boundary_candidates(rows, count=len(target_boundaries)),
            tolerance=1,
        ),
    }
    metrics["music_boundary_comparison"] = music_boundary_comparison(
        rows,
        annotation,
        struct,
        tolerance_seconds=boundary_tolerance_seconds,
    )

    song_dir = args.out_dir / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(song_dir / f"{song_id}.signal_bars.jsonl", rows)
    write_json(song_dir / f"{song_id}.signal_metrics.json", metrics)
    write_target_merged_outputs(song_dir, song_id, target_merged)
    write_merged_outputs(
        song_dir,
        song_id,
        "structure_baseline",
        target_merged,
        structure_merged,
        merged_span_comparison["structure_baseline"],
    )
    write_merged_outputs(
        song_dir,
        song_id,
        "callability_rule",
        target_merged,
        callability_merged,
        merged_span_comparison["callability_rule"],
    )
    np.save(song_dir / f"{song_id}.self_similarity.npy", self_similarity)
    plot_callability(rows, song_dir / f"{song_id}.callability_curves.png", f"{song_id}: signal features and callability")
    plot_self_similarity(self_similarity, song_dir / f"{song_id}.self_similarity.png", f"{song_id}: bar-level self-similarity")
    plot_confusion(
        metrics["callability_rule"]["confusion"],
        song_dir / f"{song_id}.callability_confusion.png",
        f"{song_id}: callability-rule confusion",
    )
    write_song_summary(song_dir / f"{song_id}.signal_summary.md", song_id, metrics)

    return {
        "metrics": metrics,
        "rows": rows,
        "y_true": y_true,
        "structure_pred": structure_pred,
        "callability_pred": callability_pred,
    }


def write_aggregate_summary(path: Path, results: List[Dict[str, Any]], aggregate: Dict[str, Any]) -> None:
    lines = [
        "# Signal Callability Experiment: Aggregate Results",
        "",
        "| Song | Bars | Structure Acc | Structure Macro-F1 | Callability Acc | Callability Macro-F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["metrics"]
        lines.append(
            "| {song} | {bars} | {sa:.3f} | {sf:.3f} | {ca:.3f} | {cf:.3f} |".format(
                song=metrics["song_id"],
                bars=metrics["bars"],
                sa=metrics["structure_baseline"]["accuracy"],
                sf=metrics["structure_baseline"]["macro_f1"],
                ca=metrics["callability_rule"]["accuracy"],
                cf=metrics["callability_rule"]["macro_f1"],
            )
        )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Method | Accuracy | Macro-F1 | Bars |",
            "|---|---:|---:|---:|",
            f"| structure_baseline | {aggregate['structure_baseline']['accuracy']:.3f} | {aggregate['structure_baseline']['macro_f1']:.3f} | {aggregate['structure_baseline']['total']} |",
            f"| callability_rule | {aggregate['callability_rule']['accuracy']:.3f} | {aggregate['callability_rule']['macro_f1']:.3f} | {aggregate['callability_rule']['total']} |",
        ]
    )
    if "loso_structure_rf" in aggregate:
        lines.append(
            f"| loso_structure_rf | {aggregate['loso_structure_rf']['accuracy']:.3f} | {aggregate['loso_structure_rf']['macro_f1']:.3f} | {aggregate['loso_structure_rf']['total']} |"
        )
        lines.append(
            f"| loso_audio_rf | {aggregate['loso_audio_rf']['accuracy']:.3f} | {aggregate['loso_audio_rf']['macro_f1']:.3f} | {aggregate['loso_audio_rf']['total']} |"
        )
    if "boundary_detection" in aggregate:
        lines.extend(["", "## Call-Role Boundary Detection", "", "| Method | Precision | Recall | F1 |", "|---|---:|---:|---:|"])
        for method in ("structure_boundary", "fused_novelty_topk"):
            item = aggregate["boundary_detection"][method]
            lines.append(f"| {method} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} |")
    if "music_boundary_comparison" in aggregate:
        lines.extend(
            [
                "",
                "## Music Segment Boundary Comparison",
                "",
                "Fine target uses all human music-section boundaries. Coarse target folds fine labels into an allin1-like label set.",
                "",
                "| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        comparison = aggregate["music_boundary_comparison"]
        for target in ("manual_fine", "manual_coarse"):
            for method in ("allin1_structure", "fused_novelty_topk"):
                item = comparison[target][method]
                lines.append(
                    f"| {target} | {method} | {item['precision']:.3f} | {item['recall']:.3f} | "
                    f"{item['f1']:.3f} | {item['target_count']} | {item['predicted_count']} |"
                )
    if "merged_span_comparison" in aggregate:
        lines.extend(
            [
                "",
                "## Merged Span Comparison",
                "",
                "| Method | Time-Weighted Role Acc | Macro Role IoU | Boundary F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in ("structure_baseline", "callability_rule", "loso_structure_rf", "loso_audio_rf"):
            item = aggregate["merged_span_comparison"].get(method)
            if not item:
                continue
            lines.append(
                f"| {method} | {item['time_weighted_role_accuracy']:.3f} | "
                f"{item['macro_role_iou']:.3f} | {item['boundary']['f1']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rows_for_loso(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for result in results:
        song_id = result["metrics"]["song_id"]
        for row in result["rows"]:
            item = dict(row)
            item["_song_id"] = song_id
            rows.append(item)
    return rows


def collect_struct_vocab(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str((row.get("features") or {}).get("allin1_struct_label") or "unknown") for row in rows})


def vectorize_row(row: Dict[str, Any], struct_vocab: Sequence[str], include_audio: bool) -> List[float]:
    features = row.get("features") or {}
    struct_label = str(features.get("allin1_struct_label") or "unknown")
    values = [
        float(features.get("relative_pos", 0.0)),
        float(features.get("duration", 0.0)),
        float(features.get("bar_duration_ratio", 0.0)),
        float(features.get("start_observed_downbeat", 0.0)),
        float(features.get("start_extrapolated_downbeat", 0.0)),
        float(features.get("allin1_struct_boundary", 0.0)),
        float(features.get("allin1_struct_label_overlap", 0.0)),
    ]
    values.extend(1.0 if struct_label == label else 0.0 for label in struct_vocab)
    if not include_audio:
        return values

    signal = row.get("signal_features") or {}
    novelty = row.get("novelty") or {}
    callability = row.get("callability") or {}
    values.extend(
        [
            float(signal.get("energy", 0.0)),
            float(signal.get("onset", 0.0)),
            float(signal.get("spectral_centroid", 0.0)),
            float(signal.get("spectral_bandwidth", 0.0)),
            float(signal.get("spectral_rolloff", 0.0)),
            float(signal.get("spectral_flatness", 0.0)),
            float(signal.get("midrange_balance", 0.0)),
            float(signal.get("vocal_density_proxy", 0.0)),
            float(signal.get("beat_stability", 0.0)),
            float(novelty.get("timbre", 0.0)),
            float(novelty.get("harmony", 0.0)),
            float(novelty.get("energy", 0.0)),
            float(novelty.get("onset", 0.0)),
            float(novelty.get("fused", 0.0)),
            float(callability.get("keepspace", 0.0)),
            float(callability.get("rhythmcall", 0.0)),
            float(callability.get("mix", 0.0)),
            float(callability.get("underground_gei", 0.0)),
        ]
    )
    values.extend(float(value) for value in signal.get("mfcc_mean", []))
    values.extend(float(value) for value in signal.get("chroma_mean", []))
    return values


def loso_random_forest(rows: Sequence[Dict[str, Any]], include_audio: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for LOSO classifier experiments.") from exc

    struct_vocab = collect_struct_vocab(rows)
    song_ids = sorted({str(row["_song_id"]) for row in rows})
    y_true_all: List[str] = []
    y_pred_all: List[str] = []
    predictions: List[Dict[str, Any]] = []

    for test_song in song_ids:
        train_rows = [row for row in rows if row["_song_id"] != test_song]
        test_rows = [row for row in rows if row["_song_id"] == test_song]
        if not train_rows or not test_rows:
            continue
        x_train = np.asarray([vectorize_row(row, struct_vocab, include_audio=include_audio) for row in train_rows], dtype=np.float32)
        y_train = [row["target"]["call_role"] for row in train_rows]
        x_test = np.asarray([vectorize_row(row, struct_vocab, include_audio=include_audio) for row in test_rows], dtype=np.float32)
        y_test = [row["target"]["call_role"] for row in test_rows]

        model = RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=7,
        )
        model.fit(x_train, y_train)
        y_pred = [normalize_role(value) for value in model.predict(x_test)]
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        for row, target, pred in zip(test_rows, y_test, y_pred):
            predictions.append(
                {
                    "song_id": row["_song_id"],
                    "bar_index": row["bar_index"],
                    "start": row["start"],
                    "end": row["end"],
                    "target": target,
                    "prediction": pred,
                    "feature_set": "audio_rf" if include_audio else "structure_rf",
                }
            )

    return classification_metrics(y_true_all, y_pred_all), predictions


def run_loso_experiments(results: Sequence[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    rows = rows_for_loso(results)
    if len({row["_song_id"] for row in rows}) < 2:
        return {}
    structure_metrics, structure_predictions = loso_random_forest(rows, include_audio=False)
    audio_metrics, audio_predictions = loso_random_forest(rows, include_audio=True)
    write_jsonl(out_dir / "loso_predictions.jsonl", structure_predictions + audio_predictions)
    prediction_maps = {
        "loso_structure_rf": {
            (item["song_id"], int(item["bar_index"])): item["prediction"]
            for item in structure_predictions
        },
        "loso_audio_rf": {
            (item["song_id"], int(item["bar_index"])): item["prediction"]
            for item in audio_predictions
        },
    }
    for result in results:
        song_id = result["metrics"]["song_id"]
        song_dir = out_dir / song_id
        rows_for_song = result["rows"]
        target_roles = [row["target"]["call_role"] for row in rows_for_song]
        target_merged = merge_role_sequence(rows_for_song, target_roles, method="target_manual_grid")
        full_bar_durations = [
            float(row["features"]["duration"])
            for row in rows_for_song
            if row.get("bar_kind") == "full_bar"
        ]
        boundary_tolerance_seconds = float(np.median(full_bar_durations)) if full_bar_durations else 2.5
        for method, mapping in prediction_maps.items():
            predicted_roles = [
                normalize_role(mapping.get((song_id, int(row["bar_index"])), "keepspace"))
                for row in rows_for_song
            ]
            predicted_merged = merge_role_sequence(rows_for_song, predicted_roles, method=method)
            comparison = merged_span_metrics(target_merged, predicted_merged, boundary_tolerance_seconds)
            result["metrics"].setdefault("merged_span_comparison", {})[method] = comparison
            write_merged_outputs(
                song_dir,
                song_id,
                method,
                target_merged,
                predicted_merged,
                comparison,
            )
        write_json(song_dir / f"{song_id}.signal_metrics.json", result["metrics"])
        write_song_summary(song_dir / f"{song_id}.signal_summary.md", song_id, result["metrics"])
    plot_confusion(
        audio_metrics["confusion"],
        out_dir / "loso_audio_rf_confusion.png",
        "LOSO audio-feature Random Forest confusion",
    )
    return {
        "loso_structure_rf": structure_metrics,
        "loso_audio_rf": audio_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build beat-synchronous signal features and callability experiments.")
    parser.add_argument("--song", help="Song id under annotations/<song>/<song>.annotation.json.")
    parser.add_argument("--annotation", type=Path, help="Explicit annotation JSON.")
    parser.add_argument("--all", action="store_true", help="Run all annotations.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--struct-dir", type=Path, default=Path("struct"))
    parser.add_argument("--struct", type=Path, help="Explicit struct JSON for a single-song run.")
    parser.add_argument("--audio", type=Path, help="Explicit audio path for a single-song run.")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    annotation_paths = resolve_annotation_paths(args)
    results = []
    for annotation_path in annotation_paths:
        print(f"Processing {annotation_path}")
        results.append(process_song(annotation_path, args, root))

    all_true: List[str] = []
    all_structure: List[str] = []
    all_callability: List[str] = []
    for result in results:
        all_true.extend(result["y_true"])
        all_structure.extend(result["structure_pred"])
        all_callability.extend(result["callability_pred"])

    aggregate = {
        "songs": len(results),
        "structure_baseline": classification_metrics(all_true, all_structure),
        "callability_rule": classification_metrics(all_true, all_callability),
        "target_distribution": role_distribution(all_true),
        "boundary_detection": aggregate_boundary_detection(results),
        "music_boundary_comparison": aggregate_music_boundary_comparison(results),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate.update(run_loso_experiments(results, args.out_dir))
    aggregate["merged_span_comparison"] = aggregate_merged_span_comparison(results)
    write_json(args.out_dir / "aggregate_signal_metrics.json", aggregate)
    write_aggregate_summary(args.out_dir / "aggregate_signal_summary.md", results, aggregate)
    plot_confusion(
        aggregate["callability_rule"]["confusion"],
        args.out_dir / "aggregate_callability_confusion.png",
        "Aggregate callability-rule confusion",
    )
    print(f"Wrote aggregate metrics: {args.out_dir / 'aggregate_signal_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
