import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class BoundaryEvidence:
    time: float
    score: float
    novelty: float
    timbre_change: float
    energy_change: float
    onset_change: float
    parent_label: str
    parent_start: float
    parent_end: float


@dataclass
class RefinedSegment:
    start: float
    end: float
    parent_label: str
    label: str
    confidence: float
    reason: str


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values, axis=-1, keepdims=True)
    std = np.nanstd(values, axis=-1, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo = np.nanmin(values)
    hi = np.nanmax(values)
    if hi - lo < 1e-6:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def time_to_frame(time: float, sr: int, hop_length: int, num_frames: int) -> int:
    frame = int(round(time * sr / hop_length))
    return int(np.clip(frame, 0, num_frames - 1))


def frame_range(start: float, end: float, sr: int, hop_length: int, num_frames: int) -> Tuple[int, int]:
    a = time_to_frame(start, sr, hop_length, num_frames)
    b = time_to_frame(end, sr, hop_length, num_frames)
    if b <= a:
        b = min(num_frames, a + 1)
    return a, b


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def local_mean(values: np.ndarray, start: float, end: float, sr: int, hop_length: int) -> float:
    a, b = frame_range(start, end, sr, hop_length, len(values))
    return float(np.mean(values[a:b]))


def local_feature_mean(features: np.ndarray, start: float, end: float, sr: int, hop_length: int) -> np.ndarray:
    a, b = frame_range(start, end, sr, hop_length, features.shape[1])
    return np.mean(features[:, a:b], axis=1)


def compute_audio_features(audio_path: Path, sr: int, hop_length: int) -> Dict[str, Any]:
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)

    min_frames = min(
        len(rms), len(onset), len(centroid), len(bandwidth), len(rolloff),
        mfcc.shape[1], chroma.shape[1],
    )
    rms = rms[:min_frames]
    onset = onset[:min_frames]
    centroid = centroid[:min_frames]
    bandwidth = bandwidth[:min_frames]
    rolloff = rolloff[:min_frames]
    mfcc = mfcc[:, :min_frames]
    chroma = chroma[:, :min_frames]

    features = np.vstack([
        zscore(rms[None, :]),
        zscore(onset[None, :]),
        zscore(centroid[None, :]),
        zscore(bandwidth[None, :]),
        zscore(rolloff[None, :]),
        zscore(mfcc),
        zscore(chroma),
    ])
    features = gaussian_filter1d(features, sigma=2.0, axis=1)

    novelty = np.concatenate([[0.0], np.linalg.norm(np.diff(features, axis=1), axis=0)])
    novelty = minmax(gaussian_filter1d(novelty, sigma=4.0))

    return {
        "sr": sr,
        "hop_length": hop_length,
        "duration": duration,
        "features": features,
        "rms": minmax(gaussian_filter1d(rms, sigma=4.0)),
        "onset": minmax(gaussian_filter1d(onset, sigma=4.0)),
        "novelty": novelty,
    }


def candidate_downbeats(downbeats: List[float], start: float, end: float, min_edge_gap: float) -> List[float]:
    return [float(t) for t in downbeats if start + min_edge_gap <= t <= end - min_edge_gap]


def boundary_score(
    t: float,
    features: np.ndarray,
    rms: np.ndarray,
    onset: np.ndarray,
    novelty: np.ndarray,
    sr: int,
    hop_length: int,
    bar_seconds: float,
    parent: Dict[str, Any],
) -> BoundaryEvidence:
    num_frames = len(novelty)
    frame = time_to_frame(t, sr, hop_length, num_frames)
    radius = max(1, int(round(0.8 * sr / hop_length)))
    novelty_score = float(np.max(novelty[max(0, frame - radius): min(num_frames, frame + radius + 1)]))

    win = max(2.0, min(8.0, bar_seconds * 2.0))
    before_start = max(parent["start"], t - win)
    before_end = max(parent["start"], t - 0.25)
    after_start = min(parent["end"], t + 0.25)
    after_end = min(parent["end"], t + win)

    before_feat = local_feature_mean(features, before_start, before_end, sr, hop_length)
    after_feat = local_feature_mean(features, after_start, after_end, sr, hop_length)
    timbre_change = float(np.linalg.norm(after_feat - before_feat) / math.sqrt(features.shape[0]))
    timbre_change = float(np.clip(timbre_change / 1.5, 0, 1))

    before_energy = local_mean(rms, before_start, before_end, sr, hop_length)
    after_energy = local_mean(rms, after_start, after_end, sr, hop_length)
    energy_change = float(np.clip(abs(after_energy - before_energy) * 1.8, 0, 1))

    before_onset = local_mean(onset, before_start, before_end, sr, hop_length)
    after_onset = local_mean(onset, after_start, after_end, sr, hop_length)
    onset_change = float(np.clip(abs(after_onset - before_onset) * 1.8, 0, 1))

    score = 0.40 * novelty_score + 0.35 * timbre_change + 0.15 * energy_change + 0.10 * onset_change
    return BoundaryEvidence(
        time=float(t),
        score=float(score),
        novelty=novelty_score,
        timbre_change=timbre_change,
        energy_change=energy_change,
        onset_change=onset_change,
        parent_label=parent["label"],
        parent_start=float(parent["start"]),
        parent_end=float(parent["end"]),
    )


def select_boundaries(
    scored: List[BoundaryEvidence],
    min_spacing: float,
    threshold: float,
    max_count: int,
) -> List[BoundaryEvidence]:
    selected: List[BoundaryEvidence] = []
    for item in sorted(scored, key=lambda x: x.score, reverse=True):
        if item.score < threshold:
            continue
        if any(abs(item.time - other.time) < min_spacing for other in selected):
            continue
        selected.append(item)
        if len(selected) >= max_count:
            break
    return sorted(selected, key=lambda x: x.time)


def nearest_scored_boundary(scored: List[BoundaryEvidence], target_time: float) -> Optional[BoundaryEvidence]:
    if not scored:
        return None
    return min(scored, key=lambda x: abs(x.time - target_time))


def relabel_piece(
    piece_start: float,
    piece_end: float,
    parent: Dict[str, Any],
    prev_parent: Optional[Dict[str, Any]],
    next_parent: Optional[Dict[str, Any]],
    is_first_piece: bool,
    is_last_piece: bool,
    confidence: float,
    audio_features: Dict[str, Any],
) -> Tuple[str, str, float]:
    parent_label = parent["label"]
    length = piece_end - piece_start
    sr = audio_features["sr"]
    hop = audio_features["hop_length"]
    rms = audio_features["rms"]

    head = local_mean(rms, piece_start, min(piece_end, piece_start + max(2.0, length * 0.35)), sr, hop)
    tail = local_mean(rms, max(piece_start, piece_end - max(2.0, length * 0.35)), piece_end, sr, hop)
    rising = tail > head + 0.06

    if parent_label == "verse" and next_parent and next_parent["label"] == "chorus" and is_last_piece:
        label = "pre_chorus_build_candidate" if rising else "pre_chorus_candidate"
        reason = "Last verse piece before chorus; check as B-melo/pre-chorus."
        if rising:
            reason = "Last verse piece before chorus with rising energy; check as ietora/pre-chorus candidate."
        return label, reason, min(0.95, confidence + (0.08 if rising else 0.0))

    if (
        parent_label == "verse"
        and prev_parent
        and prev_parent["label"] == "chorus"
        and is_first_piece
        and length <= 24.0
    ):
        return (
            "post_chorus_interlude_candidate",
            "Early verse piece after chorus; check as post-chorus interlude/tiger-fire candidate.",
            min(0.92, confidence + 0.04),
        )

    if parent_label == "verse":
        return (
            "verse_subsection",
            "Internal timbre/energy change inside allin1 verse; check as A-melo repeat or smaller verse unit.",
            confidence,
        )

    if parent_label == "chorus":
        return (
            "chorus_part",
            "Internal change inside chorus; check as chorus half or repeated sabi.",
            confidence,
        )

    if parent_label in {"inst", "solo", "break"}:
        return (
            f"{parent_label}_subsection",
            "Internal change inside instrumental/solo/break section; check for an independent call window.",
            confidence,
        )

    return parent_label, f"Kept allin1 label: {parent_label}.", max(0.35, confidence * 0.75)


def refine_segments(data: Dict[str, Any], audio_features: Dict[str, Any]) -> Tuple[List[RefinedSegment], List[BoundaryEvidence]]:
    segments = data["segments"]
    downbeats = data.get("downbeats", [])
    if len(downbeats) >= 2:
        bar_seconds = float(np.median(np.diff(downbeats)))
    else:
        bpm = data.get("bpm") or 120
        bar_seconds = 60.0 / bpm * 4.0

    boundaries: List[BoundaryEvidence] = []
    refined: List[RefinedSegment] = []
    pre_chorus_duration_ref: Optional[float] = None

    for index, parent in enumerate(segments):
        start = float(parent["start"])
        end = float(parent["end"])
        label = parent["label"]
        length = end - start
        prev_parent = segments[index - 1] if index > 0 else None
        next_parent = segments[index + 1] if index + 1 < len(segments) else None

        if label in {"start", "end"} or length < max(10.0, bar_seconds * 3.0):
            refined.append(RefinedSegment(start, end, label, label, 0.55, f"Short segment; kept allin1 label: {label}."))
            continue

        min_edge_gap = max(2.0, bar_seconds * 1.25)
        scored = [
            boundary_score(
                t,
                audio_features["features"],
                audio_features["rms"],
                audio_features["onset"],
                audio_features["novelty"],
                audio_features["sr"],
                audio_features["hop_length"],
                bar_seconds,
                parent,
            )
            for t in candidate_downbeats(downbeats, start, end, min_edge_gap)
        ]

        if length < 22.0:
            max_count = 1
        elif length < 42.0:
            max_count = 2
        else:
            max_count = 3

        min_spacing = max(6.0, bar_seconds * 2.0)
        selected = select_boundaries(scored, min_spacing=min_spacing, threshold=0.42, max_count=max_count)

        if label == "verse" and prev_parent and prev_parent["label"] == "chorus":
            early_limit = min(end, start + max(16.0, bar_seconds * 8.0))
            already_has_early = any(start < b.time <= early_limit for b in selected)
            if not already_has_early:
                early = [
                    item
                    for item in sorted(scored, key=lambda x: x.time)
                    if start + max(2.0, bar_seconds * 2.0) <= item.time <= early_limit
                    and item.score >= 0.32
                ]
                if early:
                    selected.append(early[0])
                    selected = sorted(selected, key=lambda x: x.time)
                    if len(selected) > max_count:
                        selected = selected[:max_count]

        # Repeated idol/anison forms often reuse the same pre-chorus length.
        # A strong late fill can outscore the true B-melo/pre-chorus entry, so
        # once an earlier pre-chorus length is observed, prefer a matching
        # downbeat over a very short local novelty cut before the next chorus.
        if label == "verse" and next_parent and next_parent["label"] == "chorus" and pre_chorus_duration_ref:
            latest = selected[-1] if selected else None
            current_duration = end - latest.time if latest else 0.0
            target = end - pre_chorus_duration_ref
            replacement = nearest_scored_boundary(scored, target)
            if (
                replacement
                and start + min_edge_gap <= replacement.time <= end - min_edge_gap
                and (latest is None or current_duration < pre_chorus_duration_ref * 0.65)
                and abs(replacement.time - target) <= max(bar_seconds * 1.5, 4.0)
            ):
                selected = [b for b in selected if latest is None or b.time != latest.time]
                if all(abs(replacement.time - b.time) >= min_spacing for b in selected):
                    selected.append(replacement)
                selected = sorted(selected, key=lambda x: x.time)

        if not selected:
            refined.append(RefinedSegment(start, end, label, label, 0.45, f"No strong internal boundary detected; kept allin1 label: {label}."))
            continue

        boundaries.extend(selected)
        cut_points = [start] + [b.time for b in selected] + [end]
        for piece_index, (piece_start, piece_end) in enumerate(zip(cut_points[:-1], cut_points[1:])):
            left_boundary = selected[piece_index - 1] if piece_index > 0 else None
            right_boundary = selected[piece_index] if piece_index < len(selected) else None
            evidence_score = max(
                [x.score for x in [left_boundary, right_boundary] if x is not None],
                default=0.45,
            )
            refined_label, reason, conf = relabel_piece(
                piece_start,
                piece_end,
                parent,
                prev_parent,
                next_parent,
                is_first_piece=(piece_index == 0),
                is_last_piece=(piece_index == len(cut_points) - 2),
                confidence=evidence_score,
                audio_features=audio_features,
            )
            refined.append(RefinedSegment(piece_start, piece_end, label, refined_label, conf, reason))

        if label == "verse" and next_parent and next_parent["label"] == "chorus" and selected:
            pre_chorus_duration = end - selected[-1].time
            if pre_chorus_duration >= max(10.0, bar_seconds * 4.0):
                if pre_chorus_duration_ref is None:
                    pre_chorus_duration_ref = pre_chorus_duration
                else:
                    pre_chorus_duration_ref = 0.7 * pre_chorus_duration_ref + 0.3 * pre_chorus_duration

    return refined, sorted(boundaries, key=lambda x: x.time)


def write_markdown(output_path: Path, refined: List[RefinedSegment], boundaries: List[BoundaryEvidence]) -> None:
    lines = [
        "# Secondary Audio Analysis",
        "",
        "This is an audio-evidence refinement layer over allin1. Times are candidates for human verification.",
        "",
        "## Refined Segments",
        "",
        "| Start | End | Parent | Refined Label | Confidence | Reason |",
        "|---:|---:|---|---|---:|---|",
    ]
    for seg in refined:
        lines.append(
            f"| {fmt_time(seg.start)} | {fmt_time(seg.end)} | {seg.parent_label} | "
            f"{seg.label} | {seg.confidence:.2f} | {seg.reason} |"
        )

    lines.extend([
        "",
        "## Candidate Boundaries",
        "",
        "| Time | Parent | Score | Novelty | Timbre | Energy | Onset |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ])
    for b in boundaries:
        lines.append(
            f"| {fmt_time(b.time)} | {b.parent_label} {fmt_time(b.parent_start)}-{fmt_time(b.parent_end)} | "
            f"{b.score:.2f} | {b.novelty:.2f} | {b.timbre_change:.2f} | {b.energy_change:.2f} | {b.onset_change:.2f} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct", required=True, type=Path, help="Path to allin1 JSON result.")
    parser.add_argument("--audio", type=Path, default=None, help="Path to audio. Defaults to JSON path field.")
    parser.add_argument("--out-dir", type=Path, default=Path("ouen_analysis"))
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()

    data = json.loads(args.struct.read_text(encoding="utf-8"))
    audio_path = args.audio or Path(data["path"])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    audio_features = compute_audio_features(audio_path, sr=args.sr, hop_length=args.hop_length)
    refined, boundaries = refine_segments(data, audio_features)

    result = {
        "source_struct": str(args.struct),
        "source_audio": str(audio_path),
        "method": {
            "description": "allin1 coarse segments + downbeat grid + librosa novelty/timbre/energy/onset refinement",
            "lyrics_used": False,
            "features": [
                "rms_energy",
                "onset_strength",
                "spectral_centroid",
                "spectral_bandwidth",
                "spectral_rolloff",
                "mfcc",
                "chroma",
            ],
        },
        "refined_segments": [asdict(x) for x in refined],
        "candidate_boundaries": [asdict(x) for x in boundaries],
    }

    stem = args.struct.stem
    json_path = args.out_dir / f"{stem}.secondary.json"
    md_path = args.out_dir / f"{stem}.secondary.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(md_path, refined, boundaries)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
