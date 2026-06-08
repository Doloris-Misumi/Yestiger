import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LABEL_VOCAB = [
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
    "unknown",
]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def rounded(value: float) -> float:
    return round(float(value), 3)


def estimate_bar_seconds(struct: Dict[str, Any], annotation: Dict[str, Any]) -> float:
    downbeats = [as_float(value) for value in struct.get("downbeats") or []]
    downbeats = [value for value in downbeats if value is not None]
    diffs = [b - a for a, b in zip(downbeats[:-1], downbeats[1:]) if b > a]
    if diffs:
        return float(median(diffs))

    bpm = as_float(struct.get("bpm")) or as_float((annotation.get("song") or {}).get("bpm"))
    if bpm:
        return 60.0 / bpm * 4.0
    return 2.5


def annotation_end(annotation: Dict[str, Any], struct: Dict[str, Any]) -> float:
    ends: List[float] = []
    for key in ("segments", "call_spans"):
        for item in annotation.get(key) or []:
            end = as_float(item.get("end")) if isinstance(item, dict) else None
            if end is not None:
                ends.append(end)
    for item in struct.get("segments") or []:
        end = as_float(item.get("end")) if isinstance(item, dict) else None
        if end is not None:
            ends.append(end)
    return max(ends) if ends else 0.0


def point_key(value: float) -> int:
    return int(round(value * 1000))


def add_point(points: Dict[int, Dict[str, Any]], time: float, source: str) -> None:
    key = point_key(time)
    if key not in points:
        points[key] = {"time": rounded(time), "sources": []}
    if source not in points[key]["sources"]:
        points[key]["sources"].append(source)


def build_grid(annotation: Dict[str, Any], struct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float, float]:
    song_end = annotation_end(annotation, struct)
    bar_seconds = estimate_bar_seconds(struct, annotation)
    points: Dict[int, Dict[str, Any]] = {}

    add_point(points, 0.0, "annotation_start")
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

    add_point(points, song_end, "annotation_end")
    grid = [points[key] for key in sorted(points)]
    return grid, bar_seconds, song_end


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def best_segment(
    segments: Sequence[Dict[str, Any]],
    start: float,
    end: float,
    label_key: str,
) -> Tuple[str, Optional[Dict[str, Any]], float]:
    best: Optional[Dict[str, Any]] = None
    best_overlap = 0.0
    for segment in segments:
        seg_start = as_float(segment.get("start"))
        seg_end = as_float(segment.get("end"))
        if seg_start is None or seg_end is None:
            continue
        current = overlap(start, end, seg_start, seg_end)
        if current > best_overlap:
            best = segment
            best_overlap = current
    duration = max(0.0001, end - start)
    if not best:
        return "unknown", None, 0.0
    return str(best.get(label_key) or "unknown"), best, best_overlap / duration


def boundary_starts(
    segments: Sequence[Dict[str, Any]],
    grid: Sequence[Dict[str, Any]],
    tolerance: float,
) -> Dict[int, Dict[str, Any]]:
    mapped: Dict[int, Dict[str, Any]] = {}
    grid_times = [float(item["time"]) for item in grid]
    for segment in segments:
        start = as_float(segment.get("start"))
        if start is None:
            continue
        nearest_index = min(range(len(grid_times)), key=lambda index: abs(grid_times[index] - start))
        nearest_time = grid_times[nearest_index]
        offset = start - nearest_time
        if abs(offset) <= tolerance or start == 0.0:
            mapped[nearest_index] = {
                "source_time": rounded(start),
                "offset_seconds": rounded(offset),
            }
    return mapped


def one_hot(label: str, vocab: Sequence[str]) -> List[float]:
    return [1.0 if label == item else 0.0 for item in vocab]


def build_samples(annotation: Dict[str, Any], struct: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grid, bar_seconds, song_end = build_grid(annotation, struct)
    annotation_segments = [item for item in annotation.get("segments") or [] if isinstance(item, dict)]
    struct_segments = [item for item in struct.get("segments") or [] if isinstance(item, dict)]
    boundary_tolerance = max(0.25, bar_seconds * 0.125)
    annotation_boundaries = boundary_starts(annotation_segments, grid, boundary_tolerance)
    struct_boundaries = boundary_starts(struct_segments, grid, boundary_tolerance)
    struct_vocab = sorted(
        {str(item.get("label")) for item in struct_segments if isinstance(item.get("label"), str)}
        | {"unknown"}
    )

    rows: List[Dict[str, Any]] = []
    for index in range(len(grid) - 1):
        start = float(grid[index]["time"])
        end = float(grid[index + 1]["time"])
        if end <= start:
            continue

        target_label, target_segment, target_overlap = best_segment(
            annotation_segments, start, end, "music_label"
        )
        if target_label not in LABEL_VOCAB:
            target_label = "unknown"
        struct_label, _, struct_overlap = best_segment(struct_segments, start, end, "label")
        if struct_label not in struct_vocab:
            struct_label = "unknown"

        duration = end - start
        start_sources = list(grid[index]["sources"])
        target_boundary = 1 if index in annotation_boundaries else 0
        struct_boundary = 1 if index in struct_boundaries else 0
        row = {
            "song_id": (annotation.get("song") or {}).get("song_id"),
            "bar_index": index,
            "start": rounded(start),
            "end": rounded(end),
            "bar_kind": "full_bar",
            "grid_sources": start_sources,
            "features": {
                "relative_pos": round(start / song_end, 6) if song_end else 0.0,
                "duration": rounded(duration),
                "bar_duration_ratio": round(duration / bar_seconds, 6) if bar_seconds else 0.0,
                "start_observed_downbeat": 1 if "struct_downbeat" in start_sources else 0,
                "start_extrapolated_downbeat": 1 if "extrapolated_downbeat" in start_sources else 0,
                "allin1_struct_label": struct_label,
                "allin1_struct_label_overlap": round(struct_overlap, 6),
                "allin1_struct_boundary": struct_boundary,
            },
            "target": {
                "music_label": target_label,
                "music_label_id": LABEL_VOCAB.index(target_label),
                "boundary": target_boundary,
                "label_overlap": round(target_overlap, 6),
            },
        }

        if duration < bar_seconds * 0.65:
            row["bar_kind"] = "partial_bar"
        elif duration > bar_seconds * 1.35:
            row["bar_kind"] = "long_gap"

        if index in annotation_boundaries:
            row["target"]["boundary_source_time"] = annotation_boundaries[index]["source_time"]
            row["target"]["boundary_offset_seconds"] = annotation_boundaries[index]["offset_seconds"]
        if target_segment:
            segment_start = as_float(target_segment.get("start"))
            segment_end = as_float(target_segment.get("end"))
            row["target"]["segment_start"] = rounded(segment_start if segment_start is not None else start)
            row["target"]["segment_end"] = rounded(segment_end if segment_end is not None else end)

        rows.append(row)

    feature_names = [
        "relative_pos",
        "duration",
        "bar_duration_ratio",
        "start_observed_downbeat",
        "start_extrapolated_downbeat",
        "allin1_struct_boundary",
    ] + [f"allin1_struct_label={label}" for label in struct_vocab]

    sequence = {
        "song_id": (annotation.get("song") or {}).get("song_id"),
        "title": (annotation.get("song") or {}).get("title"),
        "bpm": (annotation.get("song") or {}).get("bpm") or struct.get("bpm"),
        "bar_seconds": rounded(bar_seconds),
        "song_end": rounded(song_end),
        "label_vocab": LABEL_VOCAB,
        "struct_label_vocab": struct_vocab,
        "feature_names": feature_names,
        "bars": [
            {
                "bar_index": row["bar_index"],
                "start": row["start"],
                "end": row["end"],
                "bar_kind": row["bar_kind"],
                "grid_sources": row["grid_sources"],
            }
            for row in rows
        ],
        "x": [
            [
                row["features"]["relative_pos"],
                row["features"]["duration"],
                row["features"]["bar_duration_ratio"],
                row["features"]["start_observed_downbeat"],
                row["features"]["start_extrapolated_downbeat"],
                row["features"]["allin1_struct_boundary"],
                *one_hot(row["features"]["allin1_struct_label"], struct_vocab),
            ]
            for row in rows
        ],
        "y_label": [row["target"]["music_label"] for row in rows],
        "y_label_id": [row["target"]["music_label_id"] for row in rows],
        "y_boundary": [row["target"]["boundary"] for row in rows],
    }
    return rows, sequence


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build bar-level training samples from a YesTiger annotation.")
    parser.add_argument("song", help="Song id, e.g. mayoiuta")
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--struct", type=Path, required=True, help="Matching allin1 struct JSON path.")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/bar_level"))
    args = parser.parse_args()

    annotation_path = args.annotations_dir / args.song / f"{args.song}.annotation.json"
    annotation = load_json(annotation_path)
    struct = load_json(args.struct)
    rows, sequence = build_samples(annotation, struct)

    rows_path = args.out_dir / f"{args.song}.rows.jsonl"
    sequence_path = args.out_dir / f"{args.song}.sequence.json"
    write_jsonl(rows_path, rows)
    sequence_path.write_text(json.dumps(sequence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(rows)} bar rows: {rows_path}")
    print(f"Wrote sequence: {sequence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
