import json
import math
import re
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torchaudio


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


def load_audio(audio_path: Path, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
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


def analyze_audio(audio_path: Path, title: Optional[str] = None, job_id: Optional[str] = None) -> Dict[str, Any]:
    job_id = job_id or uuid.uuid4().hex[:12]
    song_id = slugify(title or audio_path.stem)
    y, sr, duration = load_audio(audio_path)
    bar_times, tempo = estimate_bars(y, sr, duration)
    frame_features = compute_frame_features(y, sr)
    rows = build_rows(song_id, bar_times, frame_features, sr, duration)
    call_spans = merge_role_spans(rows)

    records = load_song_records(ROOT / "experiments" / "signal_callability")
    examples = build_training_examples(records, held_out_song=song_id)
    library = load_library(ROOT / "knowledge" / "call_mix_library.json")
    record = {"song_id": song_id, "rows": rows}
    used_nonrepeatable = set()
    enriched = [
        enrich_span(
            span,
            record,
            examples,
            library,
            strategy="barfit",
            used_nonrepeatable_mix_actions=used_nonrepeatable,
        )
        for span in call_spans
    ]
    timeline = flatten_actions(enriched)
    markdown = callbook_to_markdown(song_id, enriched)
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
                "This web MVP estimates bars and structure directly from audio.",
                "Action selection uses the YesTiger knowledge library and LOSO annotation prototypes.",
            ],
        },
        "bars": rows,
        "call_spans": enriched,
        "timeline": timeline,
        "markdown": markdown,
    }


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
        "call_spans": call_spans,
        "timeline": timeline,
        "markdown": markdown_path.read_text(encoding="utf-8") if markdown_path.exists() else callbook_to_markdown(song_id, call_spans),
        "audio_path": str(audio_path) if audio_path and audio_path.exists() else None,
    }
