import json
import math
import os
import re
import shutil
import subprocess
import sys
import uuid
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio

try:
    import miniaudio
except ImportError:  # pragma: no cover - optional decoder for local uploads
    miniaudio = None


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from enrich_callbook_actions import (  # noqa: E402
    build_training_examples,
    enrich_span,
    fmt_time,
    load_library,
    load_song_records,
    safe_float,
    write_json,
)

try:
    from predict_tiny_pipeline import (  # noqa: E402
        build_rows_from_struct,
        call_rows_to_tensor,
        infer_song_end,
        make_segments,
        rows_to_tensor,
        sanitize_call_role,
        sanitize_music_label,
    )
    from train_tiny_pipeline import (  # noqa: E402
        BASE_NUMERIC_FEATURES,
        Standardizer,
        TinySequenceTagger,
    )

    def _load_json_file(path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    _PIPELINE_AVAILABLE = True
except ImportError:
    _PIPELINE_AVAILABLE = False


ROLE_VOCAB = {"keepspace", "rhythmcall", "mix", "underground_gei"}
RUN_DIR = ROOT / "webapp_runs"
UPLOAD_DIR = RUN_DIR / "uploads"
JOB_DIR = RUN_DIR / "jobs"

warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", value.strip().lower())
    cleaned = cleaned.strip("_")
    return cleaned or "uploaded_song"


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def frame_slice(start: float, end: float, sr: int, hop_length: int, num_frames: int) -> slice:
    left = int(np.clip(round(start * sr / hop_length), 0, max(0, num_frames - 1)))
    right = int(np.clip(round(end * sr / hop_length), 0, max(0, num_frames)))
    if right <= left:
        right = min(num_frames, left + 1)
    return slice(left, right)


def safe_mean(values: np.ndarray, span: slice) -> float:
    chunk = values[span]
    if chunk.size == 0:
        return 0.0
    return float(np.nanmean(chunk))


def load_audio_with_miniaudio(audio_path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    if miniaudio is None:
        raise RuntimeError("miniaudio is not installed; cannot use fallback decoder.")
    decoder_path = Path(audio_path)
    temporary_path: Optional[Path] = None
    try:
        str(decoder_path).encode("ascii")
    except UnicodeEncodeError:
        cache_dir = RUN_DIR / "_decode_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = decoder_path.suffix if decoder_path.suffix else ".audio"
        temporary_path = cache_dir / f"{uuid.uuid4().hex}{suffix}"
        shutil.copy2(decoder_path, temporary_path)
        decoder_path = temporary_path
    try:
        decoded = miniaudio.decode_file(
            str(decoder_path),
            output_format=miniaudio.SampleFormat.FLOAT32,
            nchannels=1,
            sample_rate=target_sr,
        )
    finally:
        if temporary_path:
            try:
                temporary_path.unlink()
            except OSError:
                pass
    y = np.asarray(decoded.samples, dtype=np.float32).reshape(-1)
    return y, int(decoded.sample_rate)


def load_audio(audio_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    try:
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.reshape(-1)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        y = waveform.detach().cpu().numpy().astype(np.float32)
    except Exception as exc:
        try:
            y, sr = load_audio_with_miniaudio(audio_path, target_sr)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Audio decode failed. torchaudio={type(exc).__name__}: {exc}; "
                f"miniaudio={type(fallback_exc).__name__}: {fallback_exc}"
            ) from fallback_exc
    if y.size == 0:
        raise ValueError("Uploaded audio is empty.")
    peak = float(np.max(np.abs(y)))
    if peak > 1.0:
        y = y / peak
    duration = float(len(y) / sr)
    return y, sr, duration


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1 or values.size == 0:
        return values.astype(np.float32)
    kernel = np.ones(width, dtype=np.float32) / float(width)
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def frame_audio(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if y.size < frame_length:
        padded = np.pad(y, (0, frame_length - y.size), mode="constant")
        return padded.reshape(1, frame_length)
    starts = np.arange(0, y.size - frame_length + 1, hop_length, dtype=np.int64)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int64)
    return np.stack([y[start : start + frame_length] for start in starts]).astype(np.float32)


def pick_onset_peaks(onset: np.ndarray, frame_times: np.ndarray) -> List[float]:
    if onset.size < 3:
        return []
    threshold = max(0.16, float(np.percentile(onset, 82)))
    peaks = []
    last_time = -999.0
    for index in range(1, onset.size - 1):
        if onset[index] < threshold:
            continue
        if onset[index] < onset[index - 1] or onset[index] < onset[index + 1]:
            continue
        time = float(frame_times[index])
        if time - last_time < 0.22:
            if peaks and onset[index] > onset[np.searchsorted(frame_times, peaks[-1], side="left")]:
                peaks[-1] = time
                last_time = time
            continue
        peaks.append(time)
        last_time = time
    return peaks


def estimate_tempo_from_peaks(peaks: Sequence[float]) -> float:
    if len(peaks) < 8:
        return 150.0
    diffs = np.diff(np.asarray(peaks, dtype=np.float32))
    candidate_diffs = diffs[(diffs >= 0.24) & (diffs <= 0.90)]
    if candidate_diffs.size == 0:
        return 150.0
    beat_seconds = float(np.median(candidate_diffs))
    tempo = 60.0 / max(beat_seconds, 1e-6)
    while tempo < 95.0:
        tempo *= 2.0
    while tempo > 210.0:
        tempo /= 2.0
    if tempo < 90.0 or tempo > 220.0:
        return 150.0
    return tempo


def estimate_bars(y: np.ndarray, sr: int, duration: float, hop_length: int = 1024) -> Tuple[List[float], float]:
    frame_length = 2048
    frames = frame_audio(y, frame_length, hop_length)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    energy = moving_average(minmax(rms), 5)
    onset = np.zeros_like(energy)
    if energy.size > 1:
        onset[1:] = np.maximum(0.0, np.diff(energy))
    onset = moving_average(minmax(onset), 3)
    frame_times = np.arange(energy.size, dtype=np.float32) * hop_length / float(sr)
    peaks = pick_onset_peaks(onset, frame_times)
    tempo = estimate_tempo_from_peaks(peaks)
    bar_seconds = 60.0 / tempo * 4.0

    if peaks:
        early_peaks = [peak for peak in peaks if 0.15 <= peak <= min(duration, bar_seconds * 2.0)]
        first_bar = early_peaks[0] if early_peaks and early_peaks[0] < bar_seconds * 0.7 else 0.0
    else:
        first_bar = 0.0
    count = max(2, int(math.ceil(duration / max(0.8, bar_seconds))))
    bar_times = [round(min(duration, first_bar + index * bar_seconds), 3) for index in range(count + 1)]
    bar_times = [value for value in bar_times if 0.0 <= value <= duration]
    if not bar_times or bar_times[0] > 0.15:
        bar_times.insert(0, 0.0)
    if duration - bar_times[-1] > 0.2:
        bar_times.append(round(duration, 3))
    else:
        bar_times[-1] = round(duration, 3)
    return sorted(set(bar_times)), tempo


def compute_frame_features(y: np.ndarray, sr: int, hop_length: int = 1024) -> Dict[str, np.ndarray]:
    frame_length = 2048
    frames = frame_audio(y, frame_length, hop_length)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-10)
    energy = moving_average(minmax(rms), 5)

    onset = np.zeros_like(energy)
    if energy.size > 1:
        onset[1:] = np.maximum(0.0, np.diff(energy))
    onset_norm = moving_average(minmax(onset), 3)

    signs = np.signbit(frames)
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1)
    zcr_norm = minmax(zcr)

    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frame_length), axis=1)).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_length, d=1.0 / sr).astype(np.float32)
    spectrum_sum = np.sum(spectrum, axis=1) + 1e-8
    centroid_norm = minmax(np.sum(spectrum * freqs[None, :], axis=1) / spectrum_sum)
    geometric_mean = np.exp(np.mean(np.log(spectrum + 1e-8), axis=1))
    arithmetic_mean = np.mean(spectrum, axis=1) + 1e-8
    flatness_norm = minmax(geometric_mean / arithmetic_mean)

    midrange = np.clip(1.0 - np.abs(centroid_norm - 0.45) / 0.45, 0.0, 1.0)
    vocal_density = minmax((0.42 * energy + 0.20 * (1.0 - onset_norm) + 0.18 * (1.0 - flatness_norm) + 0.12 * midrange + 0.08 * (1.0 - zcr_norm)) * energy)

    novelty = np.zeros_like(energy)
    if energy.size > 1:
        novelty[1:] = np.abs(np.diff(energy)) * 0.45 + onset_norm[1:] * 0.55
    return {
        "energy": energy,
        "onset": onset_norm,
        "centroid": centroid_norm,
        "flatness": flatness_norm,
        "vocal_density_proxy": vocal_density,
        "novelty": minmax(novelty),
        "hop_length": np.array([hop_length], dtype=np.int32),
    }


def label_for_bar(index: int, count: int, energy: float, onset: float, vocal: float) -> str:
    rel = index / max(1, count)
    if index <= 1 or rel < 0.08:
        return "intro"
    if rel > 0.94:
        return "end"
    if rel > 0.88:
        return "outro"
    if 0.50 <= rel <= 0.60 and energy > 0.45 and vocal < 0.68:
        return "instrumental_break"
    if 0.28 <= rel <= 0.38 or 0.64 <= rel <= 0.82:
        return "chorus"
    if 0.22 <= rel < 0.28 or 0.58 <= rel < 0.64:
        return "pre_chorus_build" if onset > 0.55 else "pre_chorus"
    if rel > 0.82 and energy > 0.45:
        return "post_chorus"
    return "verse"


def role_for_bar(label: str, index: int, count: int, energy: float, onset: float, vocal: float) -> str:
    if energy < 0.08 and onset < 0.12:
        return "keepspace"
    if label in {"intro", "instrumental_break", "solo", "bridge", "outro"}:
        if energy > 0.28 and (onset > 0.30 or vocal < 0.55):
            return "mix"
        return "keepspace"
    if label in {"pre_chorus_build", "post_chorus"} and onset > 0.55 and energy > 0.42:
        return "mix"
    if label == "chorus" and energy > 0.62 and vocal < 0.48 and onset > 0.58:
        return "underground_gei"
    if index >= count - 2:
        return "keepspace"
    return "rhythmcall"


def build_rows(song_id: str, bar_times: Sequence[float], features: Dict[str, np.ndarray], sr: int, duration: float) -> List[Dict[str, Any]]:
    hop_length = int(features["hop_length"][0])
    num_frames = len(features["energy"])
    full_durations = [bar_times[i + 1] - bar_times[i] for i in range(len(bar_times) - 1)]
    median_bar = float(np.median(full_durations)) if full_durations else 2.5
    rows = []
    count = max(0, len(bar_times) - 1)
    for index in range(count):
        start = float(bar_times[index])
        end = float(bar_times[index + 1])
        if end <= start:
            continue
        span = frame_slice(start, end, sr, hop_length, num_frames)
        energy = safe_mean(features["energy"], span)
        onset = safe_mean(features["onset"], span)
        vocal = safe_mean(features["vocal_density_proxy"], span)
        novelty = safe_mean(features["novelty"], span)
        label = label_for_bar(index, count, energy, onset, vocal)
        role = role_for_bar(label, index, count, energy, onset, vocal)
        duration_bar = end - start
        if duration_bar < median_bar * 0.65:
            bar_kind = "partial_bar"
        elif duration_bar > median_bar * 1.35:
            bar_kind = "long_gap"
        else:
            bar_kind = "full_bar"
        rows.append(
            {
                "song_id": song_id,
                "bar_index": index,
                "start": round(start, 3),
                "end": round(end, 3),
                "bar_kind": bar_kind,
                "grid_sources": ["estimated_downbeat" if index else "song_start"],
                "features": {
                    "relative_pos": round(start / duration, 6) if duration else 0.0,
                    "duration": round(duration_bar, 3),
                    "bar_duration_ratio": round(duration_bar / median_bar, 6) if median_bar else 1.0,
                    "start_observed_downbeat": 0,
                    "start_extrapolated_downbeat": 1,
                    "allin1_struct_label": label,
                    "allin1_struct_label_overlap": 1.0,
                    "allin1_struct_boundary": 1 if index == 0 or label != rows[-1]["target"]["music_label"] else 0,
                },
                "target": {
                    "music_label": label,
                    "music_label_id": 0,
                    "boundary": 1 if index == 0 or label != rows[-1]["target"]["music_label"] else 0,
                    "label_overlap": 1.0,
                    "segment_start": round(start, 3),
                    "segment_end": round(end, 3),
                    "call_role": role,
                    "call_overlap": 1.0,
                },
                "signal_features": {
                    "energy": round(energy, 4),
                    "onset": round(onset, 4),
                    "spectral_centroid": round(safe_mean(features["centroid"], span), 4),
                    "spectral_flatness": round(safe_mean(features["flatness"], span), 4),
                    "vocal_density_proxy": round(vocal, 4),
                    "beat_stability": 0.85 if bar_kind == "full_bar" else 0.55,
                },
                "novelty": {
                    "fused": round(novelty, 4),
                },
            }
        )
    return rows


def merge_role_spans(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in rows:
        role = str((row.get("target") or {}).get("call_role") or "keepspace")
        music = str((row.get("target") or {}).get("music_label") or "unknown")
        if current and current["call_role"] == role and current.get("music_label") == music:
            current["end"] = safe_float(row.get("end"))
            current["bar_end"] = int(row.get("bar_index", current["bar_end"]))
            current["bars"] += 1
        else:
            if current:
                spans.append(current)
            current = {
                "start": safe_float(row.get("start")),
                "end": safe_float(row.get("end")),
                "call_role": role if role in ROLE_VOCAB else "keepspace",
                "music_label": music,
                "bar_start": int(row.get("bar_index", 0)),
                "bar_end": int(row.get("bar_index", 0)),
                "bars": 1,
                "method": "web_heuristic_barfit",
            }
    if current:
        spans.append(current)

    merged: List[Dict[str, Any]] = []
    for span in spans:
        if span["call_role"] in {"mix", "underground_gei"} and span["bars"] < 1:
            span["call_role"] = "keepspace"
        if merged and merged[-1]["call_role"] == span["call_role"] and merged[-1].get("music_label") == span.get("music_label"):
            merged[-1]["end"] = span["end"]
            merged[-1]["bar_end"] = span["bar_end"]
            merged[-1]["bars"] += span["bars"]
        else:
            merged.append(span)
    return merged


def flatten_actions(call_spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for span in call_spans:
        for action in span.get("action_plan") or []:
            output.append(
                {
                    "start": safe_float(action.get("start")),
                    "end": safe_float(action.get("end")),
                    "time": str(action.get("time") or ""),
                    "action_id": str(action.get("action_id") or ""),
                    "display_name": str(action.get("display_name") or action.get("action_id") or ""),
                    "role": span.get("call_role"),
                    "music_label": span.get("music_label_context"),
                    "struct_label": span.get("allin1_struct_context"),
                    "risk": action.get("risk"),
                    "bar_count": action.get("bar_count"),
                    "typical_text": action.get("typical_text") or "",
                    "mode": action.get("mode") or "",
                }
            )
    return sorted(output, key=lambda item: (item["start"], item["end"]))


def callbook_to_markdown(song_id: str, call_spans: Sequence[Dict[str, Any]]) -> str:
    lines = [
        f"# YesTiger Web Callbook: {song_id}",
        "",
        "| Time | Role | Context | Bars | Actions | Notes |",
        "|---:|---|---|---:|---|---|",
    ]
    for span in call_spans:
        action_lines = []
        notes = []
        for action in span.get("action_plan") or []:
            risk = action.get("risk")
            risk_suffix = f" [{risk}]" if risk in {"medium", "high"} else ""
            bar_suffix = f" ({action.get('bar_count')} bars)" if action.get("bar_count") is not None else ""
            action_lines.append(f"{action.get('time')} {action.get('display_name')}{risk_suffix}{bar_suffix}")
            if action.get("typical_text"):
                notes.append(f"`{action.get('typical_text')}`")
        if not action_lines:
            action_lines.append("Keep Space")
        context = f"{span.get('music_label_context', '-')}/{span.get('allin1_struct_context', '-')}"
        lines.append(
            f"| {fmt_time(safe_float(span.get('start')))}-{fmt_time(safe_float(span.get('end')))} | "
            f"{span.get('call_role')} | {context} | {span.get('bars')} | "
            f"{'<br>'.join(action_lines)} | {'<br>'.join(notes) if notes else '-'} |"
        )
    return "\n".join(lines) + "\n"


def label_at_time(segments: Sequence[Dict[str, Any]], time: float, key: str, default: str = "-") -> str:
    for segment in segments:
        if safe_float(segment.get("start")) <= time < safe_float(segment.get("end")):
            return str(segment.get(key) or default)
    if segments and abs(time - safe_float(segments[-1].get("end"))) < 0.05:
        return str(segments[-1].get(key) or default)
    return default


def rows_to_music_segments(rows: Sequence[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for row in rows:
        target = row.get("target") or {}
        features = row.get("features") or {}
        music_label = str(target.get("music_label") or features.get("allin1_struct_label") or "unknown")
        struct_label = str(features.get("allin1_struct_label") or music_label)
        boundary = int(target.get("boundary") or features.get("allin1_struct_boundary") or 0)
        start = safe_float(row.get("start"))
        end = safe_float(row.get("end"))
        if current and current["music_label"] == music_label and not boundary:
            current["end"] = end
            current["bar_end"] = int(row.get("bar_index", current["bar_end"]))
            current["bars"] += 1
            if struct_label != current.get("struct_label"):
                current["struct_label"] = f"{current.get('struct_label')}/{struct_label}"
        else:
            if current:
                segments.append(current)
            current = {
                "start": start,
                "end": end,
                "music_label": music_label,
                "struct_label": struct_label,
                "bar_start": int(row.get("bar_index", 0)),
                "bar_end": int(row.get("bar_index", 0)),
                "bars": 1,
                "source": source,
            }
    if current:
        segments.append(current)
    return segments


def normalize_music_segments(segments: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    if not segments:
        return rows_to_music_segments(rows, source)
    normalized = []
    for segment in segments:
        start = safe_float(segment.get("start"))
        end = safe_float(segment.get("end"))
        covered = [
            row
            for row in rows
            if min(safe_float(row.get("end")), end) > max(safe_float(row.get("start")), start)
        ]
        struct_labels = [
            str((row.get("features") or {}).get("allin1_struct_label") or "")
            for row in covered
            if str((row.get("features") or {}).get("allin1_struct_label") or "")
        ]
        struct_label = Counter(struct_labels).most_common(1)[0][0] if struct_labels else str(segment.get("struct_label") or "-")
        normalized.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "music_label": str(segment.get("music_label") or segment.get("label") or "unknown"),
                "struct_label": struct_label,
                "bars": len(covered) if covered else None,
                "source": source,
                "notes": segment.get("notes") or "",
            }
        )
    return normalized


def music_segments_from_call_spans(call_spans: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pseudo_rows = []
    for index, span in enumerate(call_spans):
        music_label = str(span.get("music_label_context") or span.get("music_label") or "unknown")
        struct_label = str(span.get("allin1_struct_context") or span.get("allin1_struct_label") or music_label)
        pseudo_rows.append(
            {
                "bar_index": index,
                "start": safe_float(span.get("start")),
                "end": safe_float(span.get("end")),
                "features": {
                    "allin1_struct_label": struct_label,
                    "allin1_struct_boundary": 1,
                },
                "target": {
                    "music_label": music_label,
                    "boundary": 1,
                },
            }
        )
    return rows_to_music_segments(pseudo_rows, "stored_callbook_context")


def signal_process_summary(
    status: str,
    structure: str,
    rows_count: int,
    music_segment_count: int,
    call_span_count: int,
    timeline_count: int,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if status == "full":
        steps = [
            {"name": "Audio decode", "detail": "Load waveform, estimate duration, preserve the original audio for playback."},
            {"name": "Structure analysis", "detail": "Run allin1 to obtain section labels and downbeat grid."},
            {"name": "Bar-level features", "detail": "Convert downbeats and allin1 labels into normalized bar-wise features."},
            {"name": "Sequence models", "detail": "Tiny segmenter predicts music labels; call_slotter predicts call roles and boundaries."},
            {"name": "Action fitting", "detail": "Use barfit duration rules, context prototypes, and long-MIX non-repeat constraints."},
        ]
    else:
        steps = [
            {"name": "Audio decode", "detail": "Load waveform and estimate duration from the uploaded track."},
            {"name": "Signal features", "detail": "Use frame energy, onset strength, spectral centroid/flatness, and novelty proxies."},
            {"name": "Estimated bars", "detail": "Estimate a regular bar grid from onset peaks and tempo heuristics."},
            {"name": "Rule labels", "detail": "Assign coarse music sections and call roles from position, energy, onset, and vocal-density proxy."},
            {"name": "Action fitting", "detail": "Use the same barfit action library after the fallback segmentation."},
        ]
    summary = {
        "status": status,
        "structure": structure,
        "rows": rows_count,
        "music_segments": music_segment_count,
        "call_spans": call_span_count,
        "actions": timeline_count,
        "steps": steps,
    }
    if fallback_reason:
        summary["fallback_reason"] = fallback_reason
    return summary


def standardizer_from_json(data: Dict[str, Any]):
    """Build a Standardizer from metadata JSON. Requires pipeline imports."""
    return Standardizer(data["mean"], data["std"])


def run_allin1(audio_path: Path, output_dir: Path, timeout: int = 600) -> Dict[str, Any]:
    allin1_exe = ROOT / ".venv" / "Scripts" / "allin1.exe"
    if not allin1_exe.exists():
        raise FileNotFoundError(f"allin1 not found at {allin1_exe}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for cache_dir in (ROOT / ".cache" / "matplotlib", ROOT / ".cache" / "huggingface", ROOT / ".cache" / "torch"):
        cache_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **dict(os.environ),
        "MPLCONFIGDIR": str(ROOT / ".cache" / "matplotlib"),
        "HF_HOME": str(ROOT / ".cache" / "huggingface"),
        "TORCH_HOME": str(ROOT / ".cache" / "torch"),
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    }
    result = subprocess.run(
        [str(allin1_exe), str(audio_path), "-o", str(output_dir), "--no-multiprocess", "-d", "cpu"],
        capture_output=True, text=True, timeout=timeout, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"allin1 failed (code {result.returncode}): {result.stderr}")
    struct_files = sorted(output_dir.glob("*.json"))
    if not struct_files:
        raise FileNotFoundError(f"allin1 produced no JSON in {output_dir}")
    return _load_json_file(struct_files[0])


def load_pipeline_models(model_dir: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    label_vocab = metadata["label_vocab"]
    call_role_vocab = metadata["call_role_vocab"]
    struct_vocab = metadata["struct_vocab"]
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
    segmenter.load_state_dict(torch.load(model_dir / "segmenter.pt", map_location="cpu"))
    call_slotter.load_state_dict(torch.load(model_dir / "call_slotter.pt", map_location="cpu"))
    segmenter.eval()
    call_slotter.eval()
    return {"segmenter": segmenter, "call_slotter": call_slotter}


def _rows_to_call_spans(
    rows: Sequence[Dict[str, Any]],
    call_roles: Sequence[str],
    music_labels: Sequence[str],
    call_boundary_probs: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    call_boundary_probs = call_boundary_probs or [0.0] * len(rows)
    for index, (row, role, music_label) in enumerate(zip(rows, call_roles, music_labels)):
        features = row.get("features") or {}
        struct_label = str(features.get("allin1_struct_label") or "unknown")
        boundary_prob = float(call_boundary_probs[index]) if index < len(call_boundary_probs) else 0.0
        starts_new = (
            current is None
            or current["call_role"] != role
            or current.get("music_label") != music_label
            or (index > 0 and boundary_prob >= threshold)
        )
        if not starts_new and current:
            current["end"] = safe_float(row.get("end"))
            current["bar_end"] = int(row.get("bar_index", 0))
            current["bars"] += 1
            current["boundary_probs"].append(round(boundary_prob, 4))
        else:
            if current:
                spans.append(current)
            current = {
                "start": safe_float(row.get("start")),
                "end": safe_float(row.get("end")),
                "call_role": role,
                "music_label": music_label,
                "allin1_struct_label": struct_label,
                "bar_start": int(row.get("bar_index", 0)),
                "bar_end": int(row.get("bar_index", 0)),
                "bars": 1,
                "method": "allin1_tiny_pipeline",
                "boundary_probs": [round(boundary_prob, 4)],
            }
    if current:
        spans.append(current)
    return spans


def _analyze_audio_pipeline(
    audio_path: Path,
    song_id: str,
    title: str,
    job_id: str,
    duration: Optional[float] = None,
) -> Dict[str, Any]:
    tmp_dir = UPLOAD_DIR / f"_struct_{job_id}"
    struct = run_allin1(audio_path, tmp_dir)

    model_dir = ROOT / "models" / "tiny_pipeline"
    metadata = _load_json_file(model_dir / "metadata.json")
    models_data = load_pipeline_models(model_dir, metadata)
    base_std = standardizer_from_json(metadata["base_standardizer"])

    song_end = infer_song_end(struct, None)
    result_duration = float(duration if duration is not None else song_end)
    rows = build_rows_from_struct(struct, song_id, song_end)
    if not rows:
        raise ValueError("No bar rows could be built from allin1 struct.")

    label_vocab = metadata["label_vocab"]
    call_role_vocab = metadata["call_role_vocab"]
    struct_vocab = metadata["struct_vocab"]

    with torch.no_grad():
        seg_x = rows_to_tensor(rows, struct_vocab, base_std)
        seg_logits, seg_bnd_logits = models_data["segmenter"](seg_x)
        music_label_ids = seg_logits.argmax(dim=-1).tolist()
        music_labels = [
            sanitize_music_label(
                label_vocab[i],
                str((r.get("features") or {}).get("allin1_struct_label") or "unknown"),
            )
            for i, r in zip(music_label_ids, rows)
        ]
        seg_bnd_probs = torch.sigmoid(seg_bnd_logits).tolist()

        call_x = call_rows_to_tensor(rows, music_labels, struct_vocab, label_vocab, base_std)
        call_logits, call_bnd_logits = models_data["call_slotter"](call_x)
        call_role_ids = call_logits.argmax(dim=-1).tolist()
        call_roles = [sanitize_call_role(call_role_vocab[i]) for i in call_role_ids]
        call_bnd_probs = torch.sigmoid(call_bnd_logits).tolist()

    for index, (row, m_label, c_role) in enumerate(zip(rows, music_labels, call_roles)):
        row["target"] = {
            "music_label": m_label,
            "call_role": c_role,
            "boundary": 1 if index == 0 or float(seg_bnd_probs[index]) >= 0.5 else 0,
            "call_boundary": 1 if index == 0 or float(call_bnd_probs[index]) >= 0.5 else 0,
        }

    segments = make_segments(rows, music_labels, seg_bnd_probs, threshold=0.5)
    music_segments = normalize_music_segments(segments, rows, "tiny_segmenter")
    raw_call_spans = _rows_to_call_spans(rows, call_roles, music_labels, call_bnd_probs, threshold=0.5)

    records = load_song_records(ROOT / "experiments" / "signal_callability")
    examples = build_training_examples(records, held_out_song=song_id)
    library = load_library(ROOT / "knowledge" / "call_mix_library.json")
    record = {"song_id": song_id, "rows": rows}
    used_nonrepeatable: set = set()
    enriched = [
        enrich_span(
            span, record, examples, library,
            strategy="barfit",
            used_nonrepeatable_mix_actions=used_nonrepeatable,
        )
        for span in raw_call_spans
    ]

    timeline = flatten_actions(enriched)
    markdown = callbook_to_markdown(song_id, enriched)
    tempo = struct.get("bpm", 0)
    process = signal_process_summary(
        "full",
        "allin1_tiny_pipeline",
        len(rows),
        len(music_segments),
        len(enriched),
        len(timeline),
    )

    return {
        "job_id": job_id,
        "song": {
            "song_id": song_id,
            "title": title or audio_path.stem,
            "audio_filename": audio_path.name,
            "duration": round(result_duration, 3),
            "tempo": round(float(tempo), 2) if tempo else 0,
            "bar_count": len(rows),
        },
        "method": {
            "structure": "allin1_tiny_pipeline",
            "actions": "barfit_action",
            "notes": [
                "Uses allin1 struct + trained tiny pipeline models (segmenter/call_slotter).",
                "Action selection uses barfit with knowledge library + LOSO annotation prototypes.",
            ],
        },
        "pipeline_status": "full",
        "signal_process": process,
        "segments": segments,
        "music_segments": music_segments,
        "bars": rows,
        "call_spans": enriched,
        "timeline": timeline,
        "markdown": markdown,
    }


def _analyze_audio_heuristic(
    audio_path: Path,
    song_id: str,
    title: str,
    job_id: str,
    y: np.ndarray,
    sr: int,
    duration: float,
    fallback_reason: Optional[str] = None,
) -> Dict[str, Any]:
    bar_times, tempo = estimate_bars(y, sr, duration)
    frame_features = compute_frame_features(y, sr)
    rows = build_rows(song_id, bar_times, frame_features, sr, duration)
    call_spans = merge_role_spans(rows)
    music_segments = rows_to_music_segments(rows, "audio_heuristic")

    records = load_song_records(ROOT / "experiments" / "signal_callability")
    examples = build_training_examples(records, held_out_song=song_id)
    library = load_library(ROOT / "knowledge" / "call_mix_library.json")
    record = {"song_id": song_id, "rows": rows}
    used_nonrepeatable: set = set()
    enriched = [
        enrich_span(
            span, record, examples, library,
            strategy="barfit",
            used_nonrepeatable_mix_actions=used_nonrepeatable,
        )
        for span in call_spans
    ]
    timeline = flatten_actions(enriched)
    markdown = callbook_to_markdown(song_id, enriched)
    process = signal_process_summary(
        "fallback" if fallback_reason else "heuristic",
        "web_heuristic_estimated_bars",
        len(rows),
        len(music_segments),
        len(enriched),
        len(timeline),
        fallback_reason=fallback_reason,
    )
    return {
        "job_id": job_id,
        "song": {
            "song_id": song_id,
            "title": title or audio_path.stem,
            "audio_filename": audio_path.name,
            "duration": round(duration, 3),
            "tempo": round(float(tempo), 2),
            "bar_count": len(rows),
        },
        "method": {
            "structure": "web_heuristic_estimated_bars",
            "actions": "barfit_action",
            "notes": [
                "Fallback: crude onset-peak bar estimation + position/energy heuristics.",
                "Install allin1 and ensure models/tiny_pipeline/ exists for full pipeline analysis.",
            ],
        },
        "pipeline_status": "fallback" if fallback_reason else "heuristic",
        "signal_process": process,
        "music_segments": music_segments,
        "bars": rows,
        "call_spans": enriched,
        "timeline": timeline,
        "markdown": markdown,
    }


def analyze_audio(audio_path: Path, title: Optional[str] = None, job_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = job_id or uuid.uuid4().hex[:12]
    song_id = slugify(title or audio_path.stem)
    effective_title = title or audio_path.stem

    if _PIPELINE_AVAILABLE:
        try:
            return _analyze_audio_pipeline(audio_path, song_id, effective_title, job_id)
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            print(f"[analyzer] Pipeline failed ({fallback_reason}), falling back to heuristic.", flush=True)
            y, sr, duration = load_audio(audio_path)
            return _analyze_audio_heuristic(audio_path, song_id, effective_title, job_id, y, sr, duration, fallback_reason=fallback_reason)

    y, sr, duration = load_audio(audio_path)
    return _analyze_audio_heuristic(audio_path, song_id, effective_title, job_id, y, sr, duration)


def save_analysis_result(result: Dict[str, Any], job_dir: Path) -> Tuple[Path, Path]:
    job_dir.mkdir(parents=True, exist_ok=True)
    json_path = job_dir / "result.json"
    md_path = job_dir / "callbook.md"
    write_json(json_path, result)
    md_path.write_text(result.get("markdown", ""), encoding="utf-8")
    return json_path, md_path


def load_example_result(song_id: str) -> Dict[str, Any]:
    path = ROOT / "experiments" / "signal_callability" / song_id / f"{song_id}.merged.loso_audio_vote_rf1_logreg1_gb1.barfit_action_call_spans.json"
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    call_spans = data.get("call_spans") or []
    markdown_path = path.with_name(path.name.replace("_call_spans.json", "_callbook.md"))
    annotation_path = ROOT / "annotations" / song_id / f"{song_id}.annotation.json"
    audio_path = None
    title = song_id
    if annotation_path.exists():
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        song = annotation.get("song") or {}
        title = song.get("title") or song_id
        raw_audio = song.get("audio_path")
        if raw_audio:
            candidate = Path(str(raw_audio))
            audio_path = candidate if candidate.is_absolute() else ROOT / candidate
    timeline = flatten_actions(call_spans)
    duration = max((safe_float(span.get("end")) for span in call_spans), default=0.0)
    music_segments = music_segments_from_call_spans(call_spans)
    process = signal_process_summary(
        "stored_example",
        "stored_loso_audio_vote_rf1_logreg1_gb1",
        int(sum(int(span.get("bars") or 0) for span in call_spans)),
        len(music_segments),
        len(call_spans),
        len(timeline),
    )
    return {
        "job_id": f"example_{song_id}",
        "song": {
            "song_id": song_id,
            "title": title,
            "duration": round(duration, 3),
            "tempo": None,
            "bar_count": sum(int(span.get("bars") or 0) for span in call_spans),
        },
        "method": data.get("action_selector") or {},
        "pipeline_status": "stored_example",
        "signal_process": process,
        "music_segments": music_segments,
        "call_spans": call_spans,
        "timeline": timeline,
        "markdown": markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else callbook_to_markdown(song_id, call_spans),
        "audio_path": str(audio_path) if audio_path and audio_path.exists() else None,
    }
