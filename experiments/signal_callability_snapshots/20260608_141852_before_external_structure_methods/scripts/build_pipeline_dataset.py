import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from build_bar_level_dataset import (
    LABEL_VOCAB,
    as_float,
    best_segment,
    boundary_starts,
    build_grid,
    build_samples,
    estimate_bar_seconds,
    load_json,
    rounded,
)


CALL_ROLE_VOCAB = [
    "keepspace",
    "rhythmcall",
    "mix",
    "underground_gei",
    "unknown",
]


RISK_VALUE = {
    "low": 0.25,
    "medium": 0.6,
    "high": 1.0,
}


def normalize_key(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", text)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def load_library(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    return {
        str(action["id"]): action
        for action in data.get("actions") or []
        if isinstance(action, dict) and action.get("id")
    }


def annotation_identity_keys(annotation: Dict[str, Any], annotation_path: Path) -> List[str]:
    song = annotation.get("song") or {}
    values = [
        song.get("song_id"),
        song.get("title"),
        Path(str(song.get("audio_path") or "")).stem,
        annotation_path.parent.name,
        annotation_path.stem.replace(".annotation", ""),
    ]
    return [normalize_key(value) for value in values if normalize_key(value)]


def resolve_struct_path(annotation: Dict[str, Any], annotation_path: Path, struct_dir: Path) -> Optional[Path]:
    wanted = set(annotation_identity_keys(annotation, annotation_path))
    matches: List[Path] = []
    for path in sorted(struct_dir.glob("*.json")):
        struct = load_json(path)
        candidates = {
            normalize_key(path.stem),
            normalize_key(Path(str(struct.get("path") or "")).stem),
        }
        if wanted & candidates:
            matches.append(path)

    if len(matches) == 1:
        return matches[0]

    bpm = as_float((annotation.get("song") or {}).get("bpm"))
    if bpm is not None:
        bpm_matches = []
        for path in sorted(struct_dir.glob("*.json")):
            struct = load_json(path)
            struct_bpm = as_float(struct.get("bpm"))
            if struct_bpm is not None and abs(struct_bpm - bpm) <= 0.5:
                bpm_matches.append(path)
        if len(bpm_matches) == 1:
            return bpm_matches[0]
    return None


def resolve_call_bar_multiplier(annotation: Dict[str, Any]) -> float:
    song = annotation.get("song") if isinstance(annotation.get("song"), dict) else {}
    multiplier = as_float(song.get("call_bar_multiplier"))
    if multiplier is not None and multiplier > 0:
        return multiplier

    bpm = as_float(song.get("bpm"))
    call_bpm = as_float(song.get("call_bpm"))
    if bpm and call_bpm and bpm > 0 and call_bpm > 0:
        return call_bpm / bpm
    return 1.0


def call_bar_multiplier_for_span(span: Dict[str, Any], default_multiplier: float) -> float:
    span_multiplier = as_float(span.get("call_bar_multiplier"))
    if span_multiplier is not None and span_multiplier > 0:
        return span_multiplier
    return default_multiplier


def context_tags(label: str) -> List[str]:
    tags = {label}
    if label == "intro":
        tags.add("long_intro")
    if label in {"interlude", "instrumental_break"}:
        tags.update({"instrumental_break", "post_chorus_interlude", "high_energy_break"})
    if label == "pre_chorus_build":
        tags.update({"pre_chorus", "chorus_entry", "high_tension_gap"})
    if label == "post_chorus":
        tags.add("post_chorus_interlude")
    if label == "chorus":
        tags.update({"repeated_chorus", "chorus_entry"})
    if label == "solo":
        tags.update({"instrumental_break", "high_energy_break"})
    if label == "outro":
        tags.add("high_energy_outro")
    if label == "verse":
        tags.add("mid_energy_vocal")
    if label == "chant":
        tags.add("call_and_response")
    return sorted(tags)


def action_duration_fit(action: Dict[str, Any], bars: float) -> Tuple[float, float, float]:
    requirements = action.get("requires") or {}
    min_bars = as_float(requirements.get("min_bars"))
    max_bars = as_float(requirements.get("max_bars"))
    under = max(0.0, (min_bars or 0.0) - bars)
    over = max(0.0, bars - (max_bars if max_bars is not None else bars))
    fit = 1.0 if under <= 0.2 and over <= 0.2 else 0.0
    return fit, under, over


def action_span_feature_row(
    song_id: str,
    span_index: int,
    span: Dict[str, Any],
    action_id: str,
    action: Dict[str, Any],
    target: int,
    music_label: str,
    struct_label: str,
    span_start: float,
    span_end: float,
    song_end: float,
    duration_bars: float,
) -> Dict[str, Any]:
    tags = set(context_tags(music_label))
    best_context = set(action.get("best_context") or [])
    duration_fit, duration_under, duration_over = action_duration_fit(action, duration_bars)
    requirements = action.get("requires") or {}
    min_bars = as_float(requirements.get("min_bars"))
    max_bars = as_float(requirements.get("max_bars"))

    return {
        "group_id": f"{song_id}:{span_index}",
        "song_id": song_id,
        "span_index": span_index,
        "start": rounded(span_start),
        "end": rounded(span_end),
        "duration_bars": round(duration_bars, 6),
        "relative_pos": round(span_start / song_end, 6) if song_end else 0.0,
        "call_role": str(span.get("call_role") or "unknown"),
        "music_label": music_label,
        "allin1_struct_label": struct_label,
        "action_id": action_id,
        "target": int(target),
        "features": {
            "duration_bars": round(duration_bars, 6),
            "relative_pos": round(span_start / song_end, 6) if song_end else 0.0,
            "action_min_bars": min_bars if min_bars is not None else 0.0,
            "action_max_bars": max_bars if max_bars is not None else 64.0,
            "action_intensity": float(action.get("intensity") or 0.0),
            "action_risk": RISK_VALUE.get(str(action.get("risk") or "").lower(), 0.5),
            "duration_fit": duration_fit,
            "duration_under": round(duration_under, 6),
            "duration_over": round(duration_over, 6),
            "context_overlap": 1.0 if tags & best_context else 0.0,
        },
    }


def build_song(
    annotation_path: Path,
    struct_path: Path,
    library_actions: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    annotation = load_json(annotation_path)
    struct = load_json(struct_path)
    rows, sequence = build_samples(annotation, struct)
    grid, bar_seconds, song_end = build_grid(annotation, struct)
    default_call_bar_multiplier = resolve_call_bar_multiplier(annotation)
    grid_for_boundaries = [{"time": row["start"], "sources": row["grid_sources"]} for row in rows]
    if rows:
        grid_for_boundaries.append({"time": rows[-1]["end"], "sources": []})

    call_spans = [item for item in annotation.get("call_spans") or [] if isinstance(item, dict)]
    call_boundaries = boundary_starts(call_spans, grid_for_boundaries, max(0.25, bar_seconds * 0.125))
    song_id = str((annotation.get("song") or {}).get("song_id") or annotation_path.parent.name)

    for row in rows:
        start = float(row["start"])
        end = float(row["end"])
        call_role, call_span, call_overlap = best_segment(call_spans, start, end, "call_role")
        if not call_span or call_overlap < 0.1:
            call_role = "keepspace"
            call_span = None
            call_overlap = 0.0
        if call_role not in CALL_ROLE_VOCAB:
            call_role = "unknown"
        call_span_index = call_spans.index(call_span) if call_span in call_spans else None
        actions = list(call_span.get("recommended_actions") or []) if call_span else []
        row["target"]["call_role"] = call_role
        row["target"]["call_role_id"] = CALL_ROLE_VOCAB.index(call_role)
        row["target"]["call_boundary"] = 1 if row["bar_index"] in call_boundaries else 0
        row["target"]["call_overlap"] = round(call_overlap, 6)
        row["target"]["recommended_actions"] = actions
        row["target"]["call_span_index"] = call_span_index
        if call_span:
            call_span_start = as_float(call_span.get("start"))
            call_span_end = as_float(call_span.get("end"))
            row["target"]["call_span_start"] = rounded(call_span_start if call_span_start is not None else start)
            row["target"]["call_span_end"] = rounded(call_span_end if call_span_end is not None else end)

    sequence["call_role_vocab"] = CALL_ROLE_VOCAB
    sequence["y_call_role"] = [row["target"]["call_role"] for row in rows]
    sequence["y_call_role_id"] = [row["target"]["call_role_id"] for row in rows]
    sequence["y_call_boundary"] = [row["target"]["call_boundary"] for row in rows]
    sequence["y_actions"] = [row["target"]["recommended_actions"] for row in rows]

    action_pairs: List[Dict[str, Any]] = []
    annotation_segments = [item for item in annotation.get("segments") or [] if isinstance(item, dict)]
    struct_segments = [item for item in struct.get("segments") or [] if isinstance(item, dict)]
    for index, span in enumerate(call_spans):
        role = str(span.get("call_role") or "unknown")
        positives = {str(action_id) for action_id in span.get("recommended_actions") or []}
        if role == "keepspace" or not positives:
            continue

        start = as_float(span.get("start"))
        end = as_float(span.get("end"))
        if start is None or end is None or end <= start:
            continue

        music_label, _, _ = best_segment(annotation_segments, start, end, "music_label")
        struct_label, _, _ = best_segment(struct_segments, start, end, "label")
        music_duration_bars = max(0.0, (end - start) / bar_seconds) if bar_seconds else 0.0
        duration_bars = music_duration_bars * call_bar_multiplier_for_span(span, default_call_bar_multiplier)
        candidates = {
            action_id
            for action_id, action in library_actions.items()
            if action.get("category") == role
        } | positives

        for action_id in sorted(candidates):
            action = library_actions.get(action_id)
            if not action:
                continue
            action_pairs.append(
                action_span_feature_row(
                    song_id=song_id,
                    span_index=index,
                    span=span,
                    action_id=action_id,
                    action=action,
                    target=1 if action_id in positives else 0,
                    music_label=music_label,
                    struct_label=struct_label,
                    span_start=start,
                    span_end=end,
                    song_end=song_end,
                    duration_bars=duration_bars,
                )
            )

    summary = {
        "song_id": song_id,
        "annotation": str(annotation_path),
        "struct": str(struct_path),
        "bar_rows": len(rows),
        "action_pairs": len(action_pairs),
        "segments": len(annotation.get("segments") or []),
        "call_spans": len(call_spans),
    }
    return rows, sequence, action_pairs, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build YesTiger training data for segment, call slot, and action ranker models.")
    parser.add_argument("songs", nargs="*", help="Song ids to export. Omit with --all to export every annotation.")
    parser.add_argument("--all", action="store_true", help="Export all annotations under annotations/*/*.annotation.json.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--struct-dir", type=Path, default=Path("struct"))
    parser.add_argument("--library", type=Path, default=Path("knowledge/call_mix_library.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/pipeline"))
    args = parser.parse_args()

    if args.all:
        annotation_paths = sorted(args.annotations_dir.glob("*/*.annotation.json"))
    else:
        annotation_paths = [
            args.annotations_dir / song / f"{song}.annotation.json"
            for song in args.songs
        ]
    annotation_paths = [path for path in annotation_paths if path.exists()]
    if not annotation_paths:
        raise SystemExit("No annotation files matched.")

    library_actions = load_library(args.library)
    all_rows: List[Dict[str, Any]] = []
    all_action_pairs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    songs_dir = args.out_dir / "songs"
    songs_dir.mkdir(parents=True, exist_ok=True)

    for annotation_path in annotation_paths:
        annotation = load_json(annotation_path)
        struct_path = resolve_struct_path(annotation, annotation_path, args.struct_dir)
        if not struct_path:
            raise SystemExit(f"No matching struct found for {annotation_path}")
        rows, sequence, action_pairs, summary = build_song(annotation_path, struct_path, library_actions)
        song_id = summary["song_id"]
        write_jsonl(songs_dir / f"{song_id}.rows.jsonl", rows)
        (songs_dir / f"{song_id}.sequence.json").write_text(
            json.dumps(sequence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        all_rows.extend(rows)
        all_action_pairs.extend(action_pairs)
        summaries.append(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "bar_rows.jsonl", all_rows)
    write_jsonl(args.out_dir / "action_pairs.jsonl", all_action_pairs)
    manifest = {
        "songs": summaries,
        "label_vocab": LABEL_VOCAB,
        "call_role_vocab": CALL_ROLE_VOCAB,
        "bar_rows": len(all_rows),
        "action_pairs": len(all_action_pairs),
        "action_vocab": sorted({row["action_id"] for row in all_action_pairs}),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(all_rows)} bar rows for {len(summaries)} songs: {args.out_dir / 'bar_rows.jsonl'}")
    print(f"Wrote {len(all_action_pairs)} action pairs: {args.out_dir / 'action_pairs.jsonl'}")
    print(f"Wrote manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
