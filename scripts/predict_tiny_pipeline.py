import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torchaudio

from build_bar_level_dataset import as_float, best_segment, rounded
from build_pipeline_dataset import load_library
from train_tiny_pipeline import (
    ACTION_NUMERIC_FEATURES,
    BASE_NUMERIC_FEATURES,
    ActionRanker,
    Standardizer,
    TinySequenceTagger,
    build_base_features,
    merge_bars,
    one_hot,
    score_actions_for_span,
)


VALID_MUSIC_LABELS = {
    "intro",
    "verse",
    "pre_chorus",
    "pre_chorus_build",
    "chorus",
    "post_chorus",
    "interlude",
    "instrumental_break",
    "bridge",
    "solo",
    "outro",
    "end",
    "chant",
}

STRUCT_TO_MUSIC_LABEL = {
    "inst": "instrumental_break",
    "start": "intro",
}

TIME_EPSILON = 0.05
BAR_TOLERANCE = 0.3
STRUCTURAL_MIX_LABELS = {"bridge", "solo"}
INVALID_PATH_CHARS = '<>:"/\\|?*'


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


def safe_dir_name(value: str) -> str:
    cleaned = "".join(
        "_" if char in INVALID_PATH_CHARS or ord(char) < 32 else char
        for char in str(value or "").strip()
    ).strip(" .")
    return cleaned or "song"


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def estimate_bar_seconds(struct: Dict[str, Any]) -> float:
    downbeats = [as_float(value) for value in struct.get("downbeats") or []]
    downbeats = [value for value in downbeats if value is not None]
    diffs = [b - a for a, b in zip(downbeats[:-1], downbeats[1:]) if b > a]
    if diffs:
        return float(median(diffs))
    bpm = as_float(struct.get("bpm"))
    if bpm:
        return 60.0 / bpm * 4.0
    return 2.5


def resolve_call_bar_multiplier(struct: Dict[str, Any], call_bpm: Optional[float], call_bar_multiplier: Optional[float]) -> float:
    if call_bar_multiplier is not None:
        if call_bar_multiplier <= 0:
            raise SystemExit("--call-bar-multiplier must be greater than 0.")
        return call_bar_multiplier

    if call_bpm is not None:
        if call_bpm <= 0:
            raise SystemExit("--call-bpm must be greater than 0.")
        struct_bpm = as_float(struct.get("bpm"))
        if not struct_bpm or struct_bpm <= 0:
            raise SystemExit("--call-bpm requires a positive BPM in the struct JSON.")
        return call_bpm / struct_bpm

    return 1.0


def infer_song_end(struct: Dict[str, Any], explicit_end: Optional[float]) -> float:
    if explicit_end is not None:
        return explicit_end
    ends: List[float] = []
    for segment in struct.get("segments") or []:
        if isinstance(segment, dict):
            end = as_float(segment.get("end"))
            if end is not None:
                ends.append(end)
    for key in ("downbeats", "beats"):
        values = [as_float(value) for value in struct.get(key) or []]
        ends.extend(value for value in values if value is not None)
    return max(ends) if ends else 0.0


def point_key(value: float) -> int:
    return int(round(value * 1000))


def add_point(points: Dict[int, Dict[str, Any]], time: float, source: str) -> None:
    key = point_key(time)
    if key not in points:
        points[key] = {"time": rounded(time), "sources": []}
    if source not in points[key]["sources"]:
        points[key]["sources"].append(source)


def build_grid(struct: Dict[str, Any], song_end: float, bar_seconds: float) -> List[Dict[str, Any]]:
    points: Dict[int, Dict[str, Any]] = {}
    add_point(points, 0.0, "song_start")
    observed_downbeats = [
        value for value in (as_float(item) for item in struct.get("downbeats") or []) if value is not None
    ]
    for downbeat in observed_downbeats:
        if 0.0 < downbeat < song_end:
            add_point(points, downbeat, "struct_downbeat")

    if observed_downbeats:
        next_time = observed_downbeats[-1] + bar_seconds
        while next_time < song_end - 0.35 * bar_seconds:
            add_point(points, next_time, "extrapolated_downbeat")
            next_time += bar_seconds
    add_point(points, song_end, "song_end")
    return [points[key] for key in sorted(points)]


def nearest_boundary_indices(
    struct_segments: Sequence[Dict[str, Any]],
    grid: Sequence[Dict[str, Any]],
    tolerance: float,
) -> set:
    if not grid:
        return set()
    times = [float(item["time"]) for item in grid]
    indices = set()
    for segment in struct_segments:
        start = as_float(segment.get("start"))
        if start is None:
            continue
        nearest_index = min(range(len(times)), key=lambda index: abs(times[index] - start))
        if abs(times[nearest_index] - start) <= tolerance or start == 0.0:
            indices.add(nearest_index)
    return indices


def _compute_bar_signal_features(
    audio_path: Path,
    rows: List[Dict[str, Any]],
    target_sr: int = 16000,
    hop_length: int = 1024,
) -> None:
    """Compute per-bar energy, onset, vocal_density_proxy, beat_stability from audio and attach as row['signal_features']."""
    if not audio_path or not Path(audio_path).exists():
        return
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.reshape(-1)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    y = waveform.detach().cpu().numpy().astype(np.float32)
    if y.size == 0:
        return

    # Normalize
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak

    # Frame-level
    frame_length = 2048
    num_frames = max(0, (y.size - frame_length) // hop_length + 1)
    if num_frames < 2:
        return
    starts = np.arange(0, y.size - frame_length + 1, hop_length, dtype=np.int64)
    frames = np.stack([y[s:s+frame_length] for s in starts]).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    energy = _smooth_minmax(rms, 5)
    onset = np.zeros_like(energy)
    if energy.size > 1:
        onset[1:] = np.maximum(0.0, np.diff(energy))
    onset = _smooth_minmax(onset, 3)

    # Spectral features for vocal proxy
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0/sr).astype(np.float32)
    spec_sum = np.sum(spectrum, axis=1) + 1e-8
    centroid = np.sum(spectrum * freqs[None, :], axis=1) / spec_sum
    centroid_norm = _minmax(centroid)
    geo_mean = np.exp(np.mean(np.log(spectrum + 1e-8), axis=1))
    arith_mean = np.mean(spectrum, axis=1) + 1e-8
    flatness = _minmax(geo_mean / arith_mean)
    midrange = np.clip(1.0 - np.abs(centroid_norm - 0.45) / 0.45, 0.0, 1.0)
    vocal_density = _minmax((0.42 * energy + 0.20 * (1.0 - onset) + 0.18 * (1.0 - flatness) + 0.12 * midrange + 0.08 * 0.5) * energy)

    # Aggregate to bars
    bar_durations = [float(row["end"]) - float(row["start"]) for row in rows]
    median_bar = float(np.median(bar_durations)) if bar_durations else 2.5
    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        left = int(np.clip(round(start * sr / hop_length), 0, max(0, num_frames - 1)))
        right = int(np.clip(round(end * sr / hop_length), 0, max(0, num_frames)))
        if right <= left:
            right = min(num_frames, left + 1)
        sl = slice(left, right)
        bar_energy = float(np.mean(energy[sl])) if energy[sl].size else 0.0
        bar_onset = float(np.mean(onset[sl])) if onset[sl].size else 0.0
        bar_vocal = float(np.mean(vocal_density[sl])) if vocal_density[sl].size else 0.0
        # beat stability: combination of bar regularity and downbeat confidence
        dur = end - start
        ratio = dur / median_bar if median_bar else 1.0
        bar_reg = math.exp(-3.0 * abs(ratio - 1.0))
        grid_src = row.get("grid_sources") or []
        dq = 1.0 if "struct_downbeat" in grid_src else 0.65 if "extrapolated_downbeat" in grid_src else 0.35
        bs = min(1.0, 0.65 * bar_reg + 0.35 * dq)
        row["signal_features"] = {
            "energy": round(bar_energy, 4),
            "onset": round(bar_onset, 4),
            "vocal_density_proxy": round(bar_vocal, 4),
            "beat_stability": round(bs, 4),
        }

def _smooth_minmax(arr: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if width <= 1 or arr.size == 0:
        return _minmax(arr)
    kernel = np.ones(width, dtype=np.float32) / float(width)
    smoothed = np.convolve(arr, kernel, mode="same")
    return _minmax(smoothed)

def _minmax(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    lo = np.nanmin(arr)
    hi = np.nanmax(arr)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def build_rows_from_struct(
    struct: Dict[str, Any],
    song_id: str,
    song_end: float,
) -> List[Dict[str, Any]]:
    bar_seconds = estimate_bar_seconds(struct)
    grid = build_grid(struct, song_end, bar_seconds)
    struct_segments = [item for item in struct.get("segments") or [] if isinstance(item, dict)]
    struct_boundaries = nearest_boundary_indices(struct_segments, grid, max(0.25, bar_seconds * 0.125))
    rows: List[Dict[str, Any]] = []
    for index in range(len(grid) - 1):
        start = float(grid[index]["time"])
        end = float(grid[index + 1]["time"])
        if end <= start:
            continue
        struct_label, _, struct_overlap = best_segment(struct_segments, start, end, "label")
        duration = end - start
        bar_kind = "full_bar"
        if duration < bar_seconds * 0.65:
            bar_kind = "partial_bar"
        elif duration > bar_seconds * 1.35:
            bar_kind = "long_gap"
        rows.append(
            {
                "song_id": song_id,
                "bar_index": index,
                "start": rounded(start),
                "end": rounded(end),
                "bar_kind": bar_kind,
                "grid_sources": list(grid[index]["sources"]),
                "features": {
                    "relative_pos": round(start / song_end, 6) if song_end else 0.0,
                    "duration": rounded(duration),
                    "bar_duration_ratio": round(duration / bar_seconds, 6) if bar_seconds else 0.0,
                    "start_observed_downbeat": 1 if "struct_downbeat" in grid[index]["sources"] else 0,
                    "start_extrapolated_downbeat": 1 if "extrapolated_downbeat" in grid[index]["sources"] else 0,
                    "allin1_struct_label": str(struct_label or "unknown"),
                    "allin1_struct_label_overlap": round(struct_overlap, 6),
                    "allin1_struct_boundary": 1 if index in struct_boundaries else 0,
                },
            }
        )
    return rows


def standardizer_from_json(data: Dict[str, Any]) -> Standardizer:
    return Standardizer(data["mean"], data["std"])


def load_models(model_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    label_vocab = metadata["label_vocab"]
    call_role_vocab = metadata["call_role_vocab"]
    struct_vocab = metadata["struct_vocab"]
    action_struct_vocab = metadata["action_struct_vocab"]
    action_vocab = metadata["action_vocab"]
    hidden_dim = int(metadata["hidden_dim"])

    segmenter = TinySequenceTagger(
        input_dim=len(BASE_NUMERIC_FEATURES) + len(struct_vocab),
        hidden_dim=hidden_dim,
        num_labels=len(label_vocab),
    )
    call_slotter = TinySequenceTagger(
        input_dim=len(BASE_NUMERIC_FEATURES) + len(struct_vocab) + len(label_vocab),
        hidden_dim=hidden_dim,
        num_labels=len(call_role_vocab),
    )
    action_ranker = ActionRanker(
        input_dim=len(ACTION_NUMERIC_FEATURES)
        + len(call_role_vocab)
        + len(label_vocab)
        + len(action_struct_vocab)
        + len(action_vocab)
    )

    segmenter.load_state_dict(torch.load(model_dir / "segmenter.pt", map_location="cpu"))
    call_slotter.load_state_dict(torch.load(model_dir / "call_slotter.pt", map_location="cpu"))
    action_ranker.load_state_dict(torch.load(model_dir / "action_ranker.pt", map_location="cpu"))
    segmenter.eval()
    call_slotter.eval()
    action_ranker.eval()
    return {
        "segmenter": segmenter,
        "call_slotter": call_slotter,
        "action_ranker": action_ranker,
    }


def sanitize_music_label(label: str, struct_label: str = "unknown") -> str:
    if label in VALID_MUSIC_LABELS:
        return label
    mapped = STRUCT_TO_MUSIC_LABEL.get(struct_label)
    if mapped in VALID_MUSIC_LABELS:
        return mapped
    return "instrumental_break" if struct_label == "inst" else "verse"


def sanitize_call_role(role: str) -> str:
    return role if role in {"keepspace", "rhythmcall", "mix", "underground_gei"} else "keepspace"


def majority(values: Sequence[str], default: str = "unknown") -> str:
    if not values:
        return default
    return Counter(values).most_common(1)[0][0]


def append_note(notes: str, addition: str) -> str:
    parts = [part.strip() for part in str(notes or "").split(";") if part.strip()]
    if addition not in parts:
        parts.append(addition)
    return "; ".join(parts)


def spans_touch(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    return abs(float(left["end"]) - float(right["start"])) <= TIME_EPSILON


def merge_adjacent_music_segments(segments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for segment in segments:
        current = dict(segment)
        if (
            merged
            and spans_touch(merged[-1], current)
            and merged[-1].get("music_label") == current.get("music_label")
        ):
            merged[-1]["end"] = current["end"]
            merged[-1]["notes"] = append_note(
                str(merged[-1].get("notes") or ""),
                "postprocessed_same_music_merge",
            )
        else:
            merged.append(current)
    return merged


def action_duration_compatible(
    action: Dict[str, Any],
    duration_bars: float,
    tolerance: float = BAR_TOLERANCE,
) -> tuple[bool, str]:
    requirements = action.get("requires") or {}
    duration = action.get("duration") or {}

    allowed_bars = [
        value
        for value in (as_float(item) for item in requirements.get("allowed_bars") or [])
        if value is not None
    ]
    if allowed_bars and min(abs(duration_bars - item) for item in allowed_bars) > tolerance:
        return False, "allowed_bars"

    min_bars = as_float(requirements.get("min_bars"))
    max_bars = as_float(requirements.get("max_bars"))
    if min_bars is not None and duration_bars < min_bars - tolerance:
        return False, "under_min_bars"
    if max_bars is not None and duration_bars > max_bars + tolerance:
        return False, "over_max_bars"

    preferred_bars = as_float(duration.get("preferred_bars"))
    if duration.get("strict_bars") and preferred_bars is not None:
        can_compress = bool(duration.get("can_compress", True))
        can_extend = bool(duration.get("can_extend", True))
        if not can_compress and duration_bars < preferred_bars - tolerance:
            return False, "strict_under_preferred"
        if not can_extend and duration_bars > preferred_bars + tolerance:
            return False, "strict_over_preferred"

    if duration.get("strict_bar_multiple"):
        unit = as_float(requirements.get("bar_multiple")) or preferred_bars
        if unit:
            ratio = duration_bars / unit
            if abs(ratio - round(ratio)) > tolerance / unit:
                return False, "bar_multiple"

    return True, ""


def filter_candidates_by_duration(
    candidates: Sequence[Dict[str, Any]],
    library_actions: Dict[str, Dict[str, Any]],
    duration_bars: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    compatible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for candidate in candidates:
        action_id = str(candidate.get("action_id") or "")
        ok, reason = action_duration_compatible(library_actions.get(action_id) or {}, duration_bars)
        if ok:
            compatible.append(dict(candidate))
        else:
            item = dict(candidate)
            item["reason"] = reason
            rejected.append(item)
    return compatible, rejected


def covered_row_indices(rows: Sequence[Dict[str, Any]], start: float, end: float) -> List[int]:
    return [
        index
        for index, row in enumerate(rows)
        if min(float(row["end"]), end) > max(float(row["start"]), start)
    ]


def describe_call_span(
    span: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    music_labels: Sequence[str],
    bar_seconds: float,
) -> Dict[str, Any]:
    start = float(span["start"])
    end = float(span["end"])
    covered = covered_row_indices(rows, start, end)
    struct_label = majority(
        [str((rows[index].get("features") or {}).get("allin1_struct_label") or "unknown") for index in covered]
    )
    music_label = sanitize_music_label(
        majority([music_labels[index] for index in covered]),
        struct_label,
    )
    return {
        "start": rounded(start),
        "end": rounded(end),
        "call_role": sanitize_call_role(str(span.get("label") or span.get("call_role") or "keepspace")),
        "music_label": music_label,
        "allin1_struct_label": struct_label,
        "duration_bars": (end - start) / bar_seconds if bar_seconds else 0.0,
        "_source_spans": int(span.get("_source_spans") or 1),
    }


def should_merge_call_spans(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
    role = str(previous.get("call_role") or "")
    if role != current.get("call_role") or role not in {"mix", "underground_gei"}:
        return False
    if previous.get("music_label") != current.get("music_label") or not spans_touch(previous, current):
        return False

    combined_bars = float(previous.get("duration_bars") or 0.0) + float(current.get("duration_bars") or 0.0)
    if role == "underground_gei":
        return (
            float(previous.get("duration_bars") or 0.0) < 3.5
            and float(current.get("duration_bars") or 0.0) < 3.5
            and combined_bars <= 8.3
        )

    if previous.get("music_label") in STRUCTURAL_MIX_LABELS:
        return combined_bars <= 16.5
    return min(float(previous.get("duration_bars") or 0.0), float(current.get("duration_bars") or 0.0)) < 1.6


def merge_call_span_chunks(spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for span in spans:
        current = dict(span)
        if merged and should_merge_call_spans(merged[-1], current):
            merged[-1]["end"] = current["end"]
            merged[-1]["duration_bars"] = float(merged[-1].get("duration_bars") or 0.0) + float(
                current.get("duration_bars") or 0.0
            )
            merged[-1]["_source_spans"] = int(merged[-1].get("_source_spans") or 1) + int(
                current.get("_source_spans") or 1
            )
        else:
            merged.append(current)
    return merged


def rows_to_tensor(
    rows: Sequence[Dict[str, Any]],
    struct_vocab: Sequence[str],
    standardizer: Standardizer,
) -> torch.Tensor:
    return torch.tensor(
        [build_base_features(row, struct_vocab, standardizer) for row in rows],
        dtype=torch.float32,
    )


def call_rows_to_tensor(
    rows: Sequence[Dict[str, Any]],
    music_labels: Sequence[str],
    struct_vocab: Sequence[str],
    label_vocab: Sequence[str],
    standardizer: Standardizer,
) -> torch.Tensor:
    values = []
    for row, label in zip(rows, music_labels):
        values.append(build_base_features(row, struct_vocab, standardizer) + one_hot(label, label_vocab))
    return torch.tensor(values, dtype=torch.float32)


def make_segments(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    boundary_probs: Sequence[float],
    threshold: float,
) -> List[Dict[str, Any]]:
    raw_segments = merge_bars(rows, labels, boundary_probs, boundary_threshold=threshold)
    result = []
    for segment in raw_segments:
        result.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "music_label": sanitize_music_label(segment["label"]),
                "notes": "predicted_by_tiny_pipeline",
            }
        )
    return merge_adjacent_music_segments(result)


def make_call_spans(
    rows: Sequence[Dict[str, Any]],
    call_roles: Sequence[str],
    call_boundary_probs: Sequence[float],
    music_labels: Sequence[str],
    action_ranker: ActionRanker,
    library_actions: Dict[str, Dict[str, Any]],
    action_standardizer: Standardizer,
    call_role_vocab: Sequence[str],
    label_vocab: Sequence[str],
    action_struct_vocab: Sequence[str],
    action_vocab: Sequence[str],
    song_id: str,
    song_end: float,
    min_action_score: float,
    top_actions: int,
    threshold: float,
    call_bar_multiplier: float,
) -> List[Dict[str, Any]]:
    clean_roles = [sanitize_call_role(role) for role in call_roles]
    raw_spans = merge_bars(rows, clean_roles, call_boundary_probs, boundary_threshold=threshold)
    full_bar_durations = [float(row["features"]["duration"]) for row in rows if row.get("bar_kind") == "full_bar"]
    bar_seconds = float(np.median(full_bar_durations)) if full_bar_durations else 2.5
    described_spans = [
        describe_call_span(span, rows, music_labels, bar_seconds)
        for span in raw_spans
    ]
    merged_spans = merge_call_span_chunks(described_spans)

    bar_boundaries = sorted({float(row["start"]) for row in rows} | {float(row["end"]) for row in rows})

    def boundary_index(time: float) -> int:
        return min(range(len(bar_boundaries)), key=lambda item: abs(bar_boundaries[item] - time))

    def subspan_from(span: Dict[str, Any], start: float, end: float) -> Dict[str, Any]:
        result = dict(span)
        result["start"] = rounded(start)
        result["end"] = rounded(end)
        result["duration_bars"] = (end - start) / bar_seconds if bar_seconds else 0.0
        return result

    def score_call_span(index: int, span: Dict[str, Any], extra_notes: Sequence[str] = ()) -> Dict[str, Any]:
        role = sanitize_call_role(span["call_role"])
        music_label = str(span.get("music_label") or "unknown")
        struct_label = str(span.get("allin1_struct_label") or "unknown")
        duration = max(0.0, float(span["end"]) - float(span["start"]))
        music_duration_bars = float(span.get("duration_bars") or (duration / bar_seconds if bar_seconds else 0.0))
        call_duration_bars = music_duration_bars * call_bar_multiplier
        scoring_span = {
            "song_id": song_id,
            "span_index": index,
            "start": float(span["start"]),
            "end": float(span["end"]),
            "song_end": song_end,
            "duration_bars": call_duration_bars,
            "call_role": role,
            "music_label": music_label,
            "allin1_struct_label": struct_label,
        }
        raw_candidates = score_actions_for_span(
            action_ranker,
            library_actions,
            scoring_span,
            action_standardizer,
            call_role_vocab,
            label_vocab,
            action_struct_vocab,
            action_vocab,
            limit=None,
        )
        candidates, rejected_by_duration = filter_candidates_by_duration(
            raw_candidates,
            library_actions,
            float(scoring_span["duration_bars"]),
        )
        candidates = candidates[:top_actions]
        recommended = []
        if candidates and candidates[0]["score"] >= min_action_score:
            recommended = [candidates[0]["action_id"]]

        notes = "predicted_by_tiny_pipeline"
        if int(span.get("_source_spans") or 1) > 1:
            notes = append_note(notes, f"postprocessed_call_merge={int(span.get('_source_spans') or 1)}")
        if candidates:
            notes = append_note(
                notes,
                "candidates=" + ", ".join(f"{item['action_id']}:{item['score']:.4f}" for item in candidates[:3]),
            )
        if rejected_by_duration:
            notes = append_note(
                notes,
                "duration_rejected="
                + ", ".join(
                    f"{item['action_id']}:{item.get('reason') or 'duration'}"
                    for item in rejected_by_duration[:3]
                ),
            )
        for item in extra_notes:
            notes = append_note(notes, item)

        return {
            "start": rounded(float(span["start"])),
            "end": rounded(float(span["end"])),
            "call_role": role,
            "recommended_actions": recommended,
            "notes": notes,
            "_music_label": music_label,
            "_action_candidates": candidates,
        }

    def split_empty_long_mix_span(index: int, span: Dict[str, Any]) -> List[Dict[str, Any]]:
        base = score_call_span(index, span)
        if (
            span.get("call_role") != "mix"
            or span.get("music_label") not in STRUCTURAL_MIX_LABELS
            or base["recommended_actions"]
            or float(span.get("duration_bars") or 0.0) * call_bar_multiplier < 8.5
        ):
            return [base]

        start = float(span["start"])
        end = float(span["end"])
        cursor = start
        source = f"{start:.2f}-{end:.2f}"
        pieces: List[Dict[str, Any]] = []
        while cursor < end - TIME_EPSILON:
            start_index = boundary_index(cursor)
            if abs(bar_boundaries[start_index] - cursor) > TIME_EPSILON:
                break

            best_piece: Optional[Dict[str, Any]] = None
            candidate_music_bars = []
            for call_bars in (8, 7, 6, 4, 3, 2):
                music_bars = max(1, int(round(call_bars / call_bar_multiplier)))
                if music_bars not in candidate_music_bars:
                    candidate_music_bars.append(music_bars)

            for music_bars in candidate_music_bars:
                end_index = start_index + music_bars
                if end_index >= len(bar_boundaries):
                    continue
                candidate_end = bar_boundaries[end_index]
                if candidate_end > end + TIME_EPSILON:
                    continue
                trial = subspan_from(span, cursor, candidate_end)
                scored = score_call_span(
                    index + len(pieces),
                    trial,
                    extra_notes=(f"postprocessed_subspan_split={source}",),
                )
                if not scored["recommended_actions"]:
                    continue
                best_piece = scored
                break

            if best_piece is None:
                remainder = subspan_from(span, cursor, end)
                pieces.append(
                    score_call_span(
                        index + len(pieces),
                        remainder,
                        extra_notes=(f"postprocessed_subspan_remainder={source}",),
                    )
                )
                break

            pieces.append(best_piece)
            cursor = float(best_piece["end"])

        if len(pieces) > 1 and any(piece["recommended_actions"] for piece in pieces):
            return pieces
        return [base]

    call_spans = []
    for index, span in enumerate(merged_spans):
        call_spans.extend(split_empty_long_mix_span(index, span))
    return call_spans


def write_callbook_markdown(path: Path, callbook: Dict[str, Any]) -> None:
    lines = [
        f"# Model Callbook: {callbook['song']['title']}",
        "",
        "Generated by `scripts/predict_tiny_pipeline.py`. Treat this as a model draft and validate manually.",
        "",
        "| Time | Music | Role | Primary | Candidates |",
        "|---|---|---|---|---|",
    ]
    for entry in callbook["entries"]:
        candidates = ", ".join(f"{item['action_id']} ({item['score']:.2f})" for item in entry.get("action_candidates", [])[:3])
        primary = ", ".join(entry.get("recommended_actions") or []) or "-"
        lines.append(
            f"| {fmt_time(entry['start'])}-{fmt_time(entry['end'])} | {entry['music_label']} | "
            f"{entry['call_role']} | {primary} | {candidates or '-'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict YesTiger segments/call_spans/action candidates for a new allin1 struct.")
    parser.add_argument("--struct", required=True, type=Path, help="allin1 struct JSON.")
    parser.add_argument("--audio", type=Path, help="Optional audio path for metadata.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/tiny_pipeline"))
    parser.add_argument("--library", type=Path, default=Path("knowledge/call_mix_library.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/model_pipeline"))
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Write files directly into --out-dir instead of --out-dir/<song-id>/.",
    )
    parser.add_argument("--song-id", help="Defaults to struct stem.")
    parser.add_argument("--title", help="Defaults to audio/struct stem.")
    parser.add_argument("--artist", default="")
    parser.add_argument("--franchise", default="")
    parser.add_argument("--song-end", type=float, default=None)
    parser.add_argument("--call-bpm", type=float, default=None, help="Practical call tempo. Example: 75 for a 150 BPM half-time call grid.")
    parser.add_argument(
        "--call-bar-multiplier",
        type=float,
        default=None,
        help="Direct multiplier from analyzed music bars to practical call bars. Overrides --call-bpm.",
    )
    parser.add_argument("--segment-boundary-threshold", type=float, default=0.5)
    parser.add_argument("--call-boundary-threshold", type=float, default=0.5)
    parser.add_argument("--min-action-score", type=float, default=0.2)
    parser.add_argument("--top-actions", type=int, default=5)
    args = parser.parse_args()

    metadata = load_json(args.model_dir / "metadata.json")
    models = load_models(args.model_dir, metadata)
    base_standardizer = standardizer_from_json(metadata["base_standardizer"])
    action_standardizer = standardizer_from_json(metadata["action_standardizer"])
    library_actions = load_library(args.library)

    struct = load_json(args.struct)
    audio_path = args.audio or (Path(str(struct["path"])) if struct.get("path") else None)
    song_id = args.song_id or args.struct.stem
    title = args.title or (audio_path.stem if audio_path else args.struct.stem)
    song_end = infer_song_end(struct, args.song_end)
    call_bar_multiplier = resolve_call_bar_multiplier(struct, args.call_bpm, args.call_bar_multiplier)
    rows = build_rows_from_struct(struct, song_id, song_end)
    if not rows:
        raise SystemExit("No bar rows could be built from the struct.")

    if audio_path and Path(audio_path).exists():
        _compute_bar_signal_features(audio_path, rows)

    label_vocab = metadata["label_vocab"]
    call_role_vocab = metadata["call_role_vocab"]
    struct_vocab = metadata["struct_vocab"]
    action_struct_vocab = metadata["action_struct_vocab"]
    action_vocab = metadata["action_vocab"]

    with torch.no_grad():
        segment_x = rows_to_tensor(rows, struct_vocab, base_standardizer)
        segment_logits, segment_boundary_logits = models["segmenter"](segment_x)
        music_label_ids = segment_logits.argmax(dim=-1).tolist()
        music_labels = [sanitize_music_label(label_vocab[index], str((row.get("features") or {}).get("allin1_struct_label") or "unknown")) for index, row in zip(music_label_ids, rows)]
        segment_boundary_probs = torch.sigmoid(segment_boundary_logits).tolist()

        call_x = call_rows_to_tensor(rows, music_labels, struct_vocab, label_vocab, base_standardizer)
        call_logits, call_boundary_logits = models["call_slotter"](call_x)
        call_role_ids = call_logits.argmax(dim=-1).tolist()
        call_roles = [sanitize_call_role(call_role_vocab[index]) for index in call_role_ids]
        call_boundary_probs = torch.sigmoid(call_boundary_logits).tolist()

    segments = make_segments(
        rows,
        music_labels,
        segment_boundary_probs,
        threshold=args.segment_boundary_threshold,
    )
    call_spans_with_extras = make_call_spans(
        rows,
        call_roles,
        call_boundary_probs,
        music_labels,
        models["action_ranker"],
        library_actions,
        action_standardizer,
        call_role_vocab,
        label_vocab,
        action_struct_vocab,
        action_vocab,
        song_id,
        song_end,
        min_action_score=args.min_action_score,
        top_actions=args.top_actions,
        threshold=args.call_boundary_threshold,
        call_bar_multiplier=call_bar_multiplier,
    )

    call_spans = [
        {
            "start": span["start"],
            "end": span["end"],
            "call_role": span["call_role"],
            "recommended_actions": span["recommended_actions"],
            "notes": span["notes"],
        }
        for span in call_spans_with_extras
    ]
    callbook_entries = [
        {
            "start": span["start"],
            "end": span["end"],
            "music_label": span["_music_label"],
            "call_role": span["call_role"],
            "recommended_actions": span["recommended_actions"],
            "action_candidates": span["_action_candidates"],
        }
        for span in call_spans_with_extras
        if span["call_role"] != "keepspace"
    ]

    song = {
        "song_id": song_id,
        "title": title,
        "artist": args.artist,
        "franchise": args.franchise,
        "audio_path": str(audio_path) if audio_path else "",
        "bpm": struct.get("bpm"),
        "meter": "4/4",
    }
    if args.call_bpm is not None:
        song["call_bpm"] = args.call_bpm
    if call_bar_multiplier != 1.0:
        song["call_bar_multiplier"] = call_bar_multiplier
    annotation = {
        "annotation_version": "0.2.0",
        "song": song,
        "model_prediction": {
            "model_dir": str(args.model_dir),
            "source_struct": str(args.struct),
            "bar_rows": len(rows),
            "postprocessing": [
                "merge_adjacent_same_music_segments",
                "merge_selected_call_span_chunks",
                "filter_action_candidates_by_duration_rules",
                "split_empty_long_structural_mix_spans",
            ],
            "warning": "Machine-generated draft; manually review before using as a human annotation.",
        },
        "segments": segments,
        "call_spans": call_spans,
    }
    callbook = {
        "song": song,
        "source_struct": str(args.struct),
        "model_dir": str(args.model_dir),
        "entries": callbook_entries,
    }

    stem = song_id
    output_dir = args.out_dir if args.flat_output else args.out_dir / safe_dir_name(stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = output_dir / f"{stem}.model.annotation.json"
    callbook_json_path = output_dir / f"{stem}.model.callbook.json"
    callbook_md_path = output_dir / f"{stem}.model.callbook.md"
    bars_path = output_dir / f"{stem}.model.bars.jsonl"
    write_json(annotation_path, annotation)
    write_json(callbook_json_path, callbook)
    write_callbook_markdown(callbook_md_path, callbook)
    write_jsonl(bars_path, rows)

    print(f"Wrote {annotation_path}")
    print(f"Wrote {callbook_json_path}")
    print(f"Wrote {callbook_md_path}")
    print(f"Wrote {bars_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
