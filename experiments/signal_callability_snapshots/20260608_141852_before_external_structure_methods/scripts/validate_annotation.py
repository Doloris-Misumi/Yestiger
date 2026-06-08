import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple


ALLOWED_MUSIC_LABELS = {
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
    # Used by current human annotations and by call_mix_library contexts.
    "chant",
}

LABEL_ALIASES = {
    "inst": "instrumental_break",
    "instrumental": "instrumental_break",
}


@dataclass
class Finding:
    severity: str
    location: str
    message: str


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_key(value: Any) -> str:
    text = str(value or "").lower()
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", text)


def normalize_label(label: Any) -> str:
    text = str(label or "").strip()
    return LABEL_ALIASES.get(text, text)


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def add(finding_list: List[Finding], severity: str, location: str, message: str) -> None:
    finding_list.append(Finding(severity=severity, location=location, message=message))


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def resolve_annotation_paths(args: argparse.Namespace) -> List[Path]:
    if args.annotation:
        return [args.annotation]

    annotations_dir = args.annotations_dir
    if not args.song:
        return sorted(annotations_dir.glob("*/*.annotation.json"))

    song = args.song
    direct = Path(song)
    if direct.exists():
        return [direct]

    candidates = [
        annotations_dir / song / f"{song}.annotation.json",
        annotations_dir / f"{song}.annotation.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return [candidate]

    wanted = normalize_key(song)
    matches = []
    for path in sorted(annotations_dir.glob("**/*.annotation.json")):
        if wanted in {normalize_key(path.stem.replace(".annotation", "")), normalize_key(path.parent.name)}:
            matches.append(path)
            continue
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        song_data = data.get("song") or {}
        keys = {
            normalize_key(song_data.get("song_id")),
            normalize_key(song_data.get("title")),
            normalize_key(Path(str(song_data.get("audio_path") or "")).stem),
        }
        if wanted in keys:
            matches.append(path)

    return matches


def candidate_struct_keys(annotation: Dict[str, Any], annotation_path: Path) -> List[str]:
    song = annotation.get("song") or {}
    raw_values = [
        song.get("song_id"),
        song.get("title"),
        Path(str(song.get("audio_path") or "")).stem,
        annotation_path.parent.name,
        annotation_path.stem.replace(".annotation", ""),
    ]
    return [normalize_key(value) for value in raw_values if normalize_key(value)]


def resolve_struct_path(
    annotation: Dict[str, Any],
    annotation_path: Path,
    struct_dir: Path,
    explicit_struct: Optional[Path],
    findings: List[Finding],
) -> Optional[Path]:
    if explicit_struct:
        if explicit_struct.exists():
            return explicit_struct
        add(findings, "ERROR", "struct", f"Explicit struct file does not exist: {explicit_struct}")
        return None

    keys = set(candidate_struct_keys(annotation, annotation_path))
    struct_paths = sorted(struct_dir.glob("*.json"))
    key_matches = [
        path
        for path in struct_paths
        if normalize_key(path.stem) in keys
        or normalize_key(Path(str(safe_load_json(path).get("path") or "")).stem) in keys
    ]
    if len(key_matches) == 1:
        return key_matches[0]
    if len(key_matches) > 1:
        names = ", ".join(str(path) for path in key_matches)
        add(findings, "ERROR", "struct", f"Multiple struct files match annotation identity: {names}")
        return None

    bpm = as_float((annotation.get("song") or {}).get("bpm"))
    if bpm is not None:
        bpm_matches = []
        for path in struct_paths:
            struct = safe_load_json(path)
            struct_bpm = as_float(struct.get("bpm"))
            if struct_bpm is not None and abs(struct_bpm - bpm) <= 0.5:
                bpm_matches.append(path)
        if len(bpm_matches) == 1:
            add(findings, "WARN", "struct", f"Struct auto-matched by BPM only: {bpm_matches[0]}")
            return bpm_matches[0]
        if len(bpm_matches) > 1:
            names = ", ".join(str(path) for path in bpm_matches)
            add(findings, "WARN", "struct", f"Multiple struct files share BPM {bpm:g}: {names}")

    add(findings, "WARN", "struct", "No matching struct file found; struct alignment checks skipped.")
    return None


def safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        return load_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def nearest_value(values: Sequence[float], target: float) -> Tuple[Optional[float], Optional[int], float]:
    if not values:
        return None, None, float("inf")
    best_index, best_value = min(enumerate(values), key=lambda item: abs(item[1] - target))
    return best_value, best_index, abs(best_value - target)


def estimate_bar_seconds(struct: Optional[Dict[str, Any]]) -> float:
    downbeats = (struct or {}).get("downbeats") or []
    if len(downbeats) >= 2:
        diffs = [b - a for a, b in zip(downbeats[:-1], downbeats[1:]) if b > a]
        if diffs:
            return float(median(diffs))

    bpm = as_float((struct or {}).get("bpm"))
    if bpm:
        return 60.0 / bpm * 4.0
    return 2.5


def estimate_bars(start: float, end: float, struct: Optional[Dict[str, Any]], tolerance: float) -> float:
    downbeats = [float(value) for value in (struct or {}).get("downbeats") or [] if as_float(value) is not None]
    duration_bars = max(0.0, (end - start) / estimate_bar_seconds(struct))
    if downbeats:
        nearest_start, start_index, start_delta = nearest_value(downbeats, start)
        nearest_end, end_index, end_delta = nearest_value(downbeats, end)
        if (
            nearest_start is not None
            and nearest_end is not None
            and start_index is not None
            and end_index is not None
            and start_delta <= tolerance
            and end_delta <= tolerance
            and end_index >= start_index
        ):
            snapped_bars = float(end_index - start_index)
            if abs(snapped_bars - duration_bars) <= max(1.0, duration_bars * 0.25):
                return snapped_bars

    return duration_bars


def find_struct_overlap(
    start: float,
    end: float,
    struct_segments: Sequence[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], float]:
    best_segment = None
    best_overlap = 0.0
    for segment in struct_segments:
        seg_start = as_float(segment.get("start"))
        seg_end = as_float(segment.get("end"))
        if seg_start is None or seg_end is None:
            continue
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment = segment
    duration = max(0.0001, end - start)
    return best_segment, best_overlap / duration


def context_tags(label: str, parent_label: Optional[str], duration: float) -> List[str]:
    label = normalize_label(label)
    parent_label = normalize_label(parent_label or label)
    tags = {label, parent_label}

    if label == "intro" and duration >= 16:
        tags.add("long_intro")
    if label in {"interlude", "instrumental_break"}:
        tags.update({"instrumental_break", "post_chorus_interlude"})
        if duration >= 20:
            tags.add("extended_instrumental")
    if label == "pre_chorus_build":
        tags.update({"pre_chorus", "chorus_entry", "high_tension_gap"})
    if label == "post_chorus":
        tags.add("post_chorus_interlude")
    if label == "chorus":
        tags.update({"repeated_chorus", "chorus_entry"})
    if label == "verse":
        tags.add("mid_energy_vocal")
    if label == "solo":
        tags.update({"instrumental_break", "emotional_solo", "member_focus"})
    if label == "chant":
        tags.update({"vocal_chant", "call_and_response", "short_vocal_hook"})
    if label == "outro":
        tags.add("high_energy_outro")

    return sorted(tags)


def validate_annotation_shape(annotation: Dict[str, Any], path: Path, findings: List[Finding]) -> None:
    if not isinstance(annotation, dict):
        add(findings, "ERROR", str(path), "Annotation root must be a JSON object.")
        return

    song = annotation.get("song")
    if not isinstance(song, dict):
        add(findings, "ERROR", "song", "Missing or invalid song object.")
    else:
        for field in ["song_id", "title", "audio_path", "bpm", "meter"]:
            if field not in song:
                add(findings, "WARN", f"song.{field}", "Missing song metadata field.")

        audio_path_text = song.get("audio_path")
        if isinstance(audio_path_text, str) and audio_path_text:
            audio_path = Path(audio_path_text)
            if not audio_path.is_absolute():
                audio_path = Path.cwd() / audio_path
            if not audio_path.exists():
                add(findings, "WARN", "song.audio_path", f"Audio path does not exist: {song.get('audio_path')}")

    segments = annotation.get("segments")
    if not isinstance(segments, list) or not segments:
        add(findings, "ERROR", "segments", "Missing or empty segments list.")


def validate_segments(
    annotation: Dict[str, Any],
    struct: Optional[Dict[str, Any]],
    library_actions: Dict[str, Dict[str, Any]],
    tolerance: float,
    findings: List[Finding],
) -> None:
    segments = annotation.get("segments") or []
    if not isinstance(segments, list):
        return

    categories = {action.get("category") for action in library_actions.values() if action.get("category")}
    allowed_roles = categories | {"keepspace"}
    struct_segments = (struct or {}).get("segments") or []
    downbeats = [float(value) for value in (struct or {}).get("downbeats") or [] if as_float(value) is not None]
    split_call_schema = isinstance(annotation.get("call_spans"), list)
    song = annotation.get("song") if isinstance(annotation.get("song"), dict) else {}
    default_call_bar_multiplier = as_float(song.get("call_bar_multiplier"))
    if default_call_bar_multiplier is None:
        default_call_bar_multiplier = 1.0
    elif default_call_bar_multiplier <= 0:
        add(findings, "ERROR", "song.call_bar_multiplier", "call_bar_multiplier must be greater than 0.")
        default_call_bar_multiplier = 1.0
    previous_end: Optional[float] = None

    for index, segment in enumerate(segments):
        location = f"segments[{index}]"
        if not isinstance(segment, dict):
            add(findings, "ERROR", location, "Segment must be a JSON object.")
            continue

        start = as_float(segment.get("start"))
        end = as_float(segment.get("end"))
        if start is None or end is None:
            add(findings, "ERROR", location, "Segment start/end must be numbers.")
            continue
        if start < 0:
            add(findings, "ERROR", location, f"Segment starts before 0: {start:g}")
        if end <= start:
            add(findings, "ERROR", location, f"Segment end must be after start: {start:g} -> {end:g}")
            continue

        call_bar_multiplier = default_call_bar_multiplier
        segment_multiplier = as_float(segment.get("call_bar_multiplier"))
        if segment_multiplier is not None:
            if segment_multiplier <= 0:
                add(findings, "ERROR", f"{location}.call_bar_multiplier", "call_bar_multiplier must be greater than 0.")
            else:
                call_bar_multiplier = segment_multiplier

        if previous_end is None:
            if abs(start) > tolerance:
                add(findings, "WARN", location, f"First segment starts at {start:g}, not 0.0.")
        else:
            gap = start - previous_end
            if abs(gap) > tolerance:
                kind = "gap" if gap > 0 else "overlap"
                add(findings, "ERROR", location, f"Timeline {kind} versus previous segment: {gap:+.3f}s.")
        previous_end = end

        label = normalize_label(segment.get("music_label"))
        if not label:
            add(findings, "ERROR", f"{location}.music_label", "Missing music_label.")
        elif label not in ALLOWED_MUSIC_LABELS:
            add(findings, "WARN", f"{location}.music_label", f"Unknown music_label: {label}")

        segment_has_call_data = "call_role" in segment or "recommended_actions" in segment
        call_role = str(segment.get("call_role") or "").strip()
        if not (split_call_schema and not segment_has_call_data):
            if not call_role:
                add(findings, "ERROR", f"{location}.call_role", "Missing call_role.")
            elif call_role not in allowed_roles:
                add(findings, "ERROR", f"{location}.call_role", f"Unknown call_role: {call_role}")

        if downbeats:
            for field_name, value in [("start", start), ("end", end)]:
                if index == 0 and field_name == "start" and abs(value) <= tolerance:
                    continue
                nearest, _, delta = nearest_value(downbeats, value)
                if delta > tolerance:
                    add(
                        findings,
                        "WARN",
                        f"{location}.{field_name}",
                        f"{fmt_time(value)} is {delta:.3f}s from nearest downbeat"
                        + (f" ({fmt_time(nearest)})" if nearest is not None else "."),
                    )

        parent_label = None
        if struct_segments:
            parent, overlap_ratio = find_struct_overlap(start, end, struct_segments)
            if parent is None or overlap_ratio < 0.75:
                add(findings, "WARN", location, "Segment does not mostly fit inside one struct segment.")
            else:
                parent_label = normalize_label(parent.get("label"))
                if label and parent_label and label != parent_label:
                    add(
                        findings,
                        "INFO",
                        f"{location}.music_label",
                        f"Annotation label '{label}' refines/differs from struct label '{parent_label}'.",
                    )

        if split_call_schema and not segment_has_call_data:
            continue

        recommended_actions = segment.get("recommended_actions", [])
        if recommended_actions is None:
            recommended_actions = []
        if not isinstance(recommended_actions, list):
            add(findings, "ERROR", f"{location}.recommended_actions", "recommended_actions must be a list.")
            continue

        seen_actions = set()
        for action_id in recommended_actions:
            if not isinstance(action_id, str):
                add(findings, "ERROR", f"{location}.recommended_actions", "Action ids must be strings.")
                continue
            if action_id in seen_actions:
                add(findings, "WARN", f"{location}.recommended_actions", f"Duplicate action id: {action_id}")
            seen_actions.add(action_id)

            action = library_actions.get(action_id)
            if not action:
                add(findings, "ERROR", f"{location}.recommended_actions", f"Unknown action id: {action_id}")
                continue

            action_category = action.get("category")
            if call_role and action_category and call_role != action_category:
                add(
                    findings,
                    "ERROR",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' category '{action_category}' does not match call_role '{call_role}'.",
                )

            music_bars = estimate_bars(start, end, struct, tolerance)
            bars = music_bars * call_bar_multiplier
            requirements = action.get("requires") or {}
            min_bars = as_float(requirements.get("min_bars"))
            max_bars = as_float(requirements.get("max_bars"))
            if min_bars is not None and bars + 0.15 < min_bars:
                add(
                    findings,
                    "WARN",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' may be too short: {bars:.2f} call bars < min {min_bars:g}.",
                )
            if max_bars is not None and bars - 0.15 > max_bars:
                add(
                    findings,
                    "WARN",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' may be too long: {bars:.2f} call bars > max {max_bars:g}.",
                )

            tags = set(context_tags(label, parent_label, end - start))
            best_context = set(action.get("best_context") or [])
            if best_context and not tags & best_context:
                add(
                    findings,
                    "INFO",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' has no best_context overlap with {sorted(tags)}.",
                )

        if call_role and call_role != "keepspace" and not recommended_actions:
            add(findings, "WARN", f"{location}.recommended_actions", f"call_role '{call_role}' has no actions.")
        if call_role == "keepspace" and recommended_actions:
            add(findings, "WARN", f"{location}.recommended_actions", "keepspace segment has actions.")

    if struct:
        struct_end = None
        if struct_segments:
            struct_end = as_float(struct_segments[-1].get("end"))
        if struct_end is None:
            beats = [as_float(value) for value in struct.get("beats") or []]
            beat_values = [value for value in beats if value is not None]
            if beat_values:
                struct_end = max(beat_values)
        if struct_end is not None and previous_end is not None and abs(previous_end - struct_end) > max(0.5, tolerance):
            add(
                findings,
                "WARN",
                "segments[-1].end",
                f"Annotation ends at {fmt_time(previous_end)}, struct ends near {fmt_time(struct_end)}.",
            )


def music_context_tags_for_span(annotation: Dict[str, Any], start: float, end: float) -> List[str]:
    tags = set()
    for segment in annotation.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        seg_start = as_float(segment.get("start"))
        seg_end = as_float(segment.get("end"))
        if seg_start is None or seg_end is None:
            continue
        overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
        if overlap <= 0:
            continue
        label = normalize_label(segment.get("music_label"))
        if label:
            tags.update(context_tags(label, label, seg_end - seg_start))
    return sorted(tags)


def validate_call_spans(
    annotation: Dict[str, Any],
    struct: Optional[Dict[str, Any]],
    library_actions: Dict[str, Dict[str, Any]],
    tolerance: float,
    findings: List[Finding],
) -> None:
    call_spans = annotation.get("call_spans")
    if call_spans is None:
        return
    if not isinstance(call_spans, list):
        add(findings, "ERROR", "call_spans", "call_spans must be a list when present.")
        return

    categories = {action.get("category") for action in library_actions.values() if action.get("category")}
    allowed_roles = categories | {"keepspace"}
    song = annotation.get("song") if isinstance(annotation.get("song"), dict) else {}
    default_call_bar_multiplier = as_float(song.get("call_bar_multiplier"))
    if default_call_bar_multiplier is None or default_call_bar_multiplier <= 0:
        default_call_bar_multiplier = 1.0

    for index, span in enumerate(call_spans):
        location = f"call_spans[{index}]"
        if not isinstance(span, dict):
            add(findings, "ERROR", location, "Call span must be a JSON object.")
            continue

        start = as_float(span.get("start"))
        end = as_float(span.get("end"))
        if start is None or end is None:
            add(findings, "ERROR", location, "Call span start/end must be numbers.")
            continue
        if start < 0:
            add(findings, "ERROR", location, f"Call span starts before 0: {start:g}")
        if end <= start:
            add(findings, "ERROR", location, f"Call span end must be after start: {start:g} -> {end:g}")
            continue

        call_role = str(span.get("call_role") or "").strip()
        if not call_role:
            add(findings, "ERROR", f"{location}.call_role", "Missing call_role.")
        elif call_role not in allowed_roles:
            add(findings, "ERROR", f"{location}.call_role", f"Unknown call_role: {call_role}")

        recommended_actions = span.get("recommended_actions", [])
        if recommended_actions is None:
            recommended_actions = []
        if not isinstance(recommended_actions, list):
            add(findings, "ERROR", f"{location}.recommended_actions", "recommended_actions must be a list.")
            continue

        call_bar_multiplier = default_call_bar_multiplier
        span_multiplier = as_float(span.get("call_bar_multiplier"))
        if span_multiplier is not None:
            if span_multiplier <= 0:
                add(findings, "ERROR", f"{location}.call_bar_multiplier", "call_bar_multiplier must be greater than 0.")
            else:
                call_bar_multiplier = span_multiplier

        tags = set(music_context_tags_for_span(annotation, start, end))
        seen_actions = set()
        for action_id in recommended_actions:
            if not isinstance(action_id, str):
                add(findings, "ERROR", f"{location}.recommended_actions", "Action ids must be strings.")
                continue
            if action_id in seen_actions:
                add(findings, "WARN", f"{location}.recommended_actions", f"Duplicate action id: {action_id}")
            seen_actions.add(action_id)

            action = library_actions.get(action_id)
            if not action:
                add(findings, "ERROR", f"{location}.recommended_actions", f"Unknown action id: {action_id}")
                continue

            action_category = action.get("category")
            if call_role and action_category and call_role != action_category:
                add(
                    findings,
                    "ERROR",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' category '{action_category}' does not match call_role '{call_role}'.",
                )

            bars = estimate_bars(start, end, struct, tolerance) * call_bar_multiplier
            requirements = action.get("requires") or {}
            min_bars = as_float(requirements.get("min_bars"))
            max_bars = as_float(requirements.get("max_bars"))
            if min_bars is not None and bars + 0.15 < min_bars:
                add(
                    findings,
                    "WARN",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' may be too short: {bars:.2f} call bars < min {min_bars:g}.",
                )
            if max_bars is not None and bars - 0.15 > max_bars:
                add(
                    findings,
                    "WARN",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' may be too long: {bars:.2f} call bars > max {max_bars:g}.",
                )

            best_context = set(action.get("best_context") or [])
            if tags and best_context and not tags & best_context:
                add(
                    findings,
                    "INFO",
                    f"{location}.recommended_actions",
                    f"Action '{action_id}' has no best_context overlap with {sorted(tags)}.",
                )

        if call_role and call_role != "keepspace" and not recommended_actions:
            add(findings, "WARN", f"{location}.recommended_actions", f"call_role '{call_role}' has no actions.")
        if call_role == "keepspace" and recommended_actions:
            add(findings, "WARN", f"{location}.recommended_actions", "keepspace span has actions.")


def load_library(path: Path, findings: List[Finding]) -> Dict[str, Dict[str, Any]]:
    try:
        library = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        add(findings, "ERROR", "library", f"Could not load library {path}: {exc}")
        return {}

    actions = library.get("actions")
    if not isinstance(actions, list):
        add(findings, "ERROR", "library.actions", "Library actions must be a list.")
        return {}

    by_id: Dict[str, Dict[str, Any]] = {}
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or not action.get("id"):
            add(findings, "ERROR", f"library.actions[{index}]", "Action must be an object with an id.")
            continue
        action_id = action["id"]
        if action_id in by_id:
            add(findings, "ERROR", f"library.actions[{index}]", f"Duplicate action id: {action_id}")
        by_id[action_id] = action
    return by_id


def print_report(annotation_path: Path, struct_path: Optional[Path], findings: Sequence[Finding]) -> None:
    print(f"Annotation: {annotation_path}")
    print(f"Struct: {struct_path if struct_path else '(not found)'}")
    print("")

    severity_order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for finding in sorted(findings, key=lambda item: (severity_order.get(item.severity, 9), item.location)):
        print(f"{finding.severity}: {finding.location}: {finding.message}")

    if findings:
        print("")
    counts = {severity: sum(1 for item in findings if item.severity == severity) for severity in ["ERROR", "WARN", "INFO"]}
    print(f"Summary: {counts['ERROR']} error(s), {counts['WARN']} warning(s), {counts['INFO']} info item(s).")


def validate_one(annotation_path: Path, args: argparse.Namespace) -> Tuple[List[Finding], Optional[Path]]:
    findings: List[Finding] = []
    try:
        annotation = load_json(annotation_path)
    except (OSError, json.JSONDecodeError) as exc:
        return [Finding("ERROR", str(annotation_path), f"Could not load annotation: {exc}")], None

    validate_annotation_shape(annotation, annotation_path, findings)
    library_actions = load_library(args.library, findings)
    struct_path = resolve_struct_path(annotation, annotation_path, args.struct_dir, args.struct, findings)
    struct = load_json(struct_path) if struct_path else None
    validate_segments(annotation, struct, library_actions, args.tolerance, findings)
    validate_call_spans(annotation, struct, library_actions, args.tolerance, findings)
    return findings, struct_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate YesTiger annotation JSON against struct and call library.")
    parser.add_argument("song", nargs="?", help="Song id/name, annotation path, or omitted to validate all annotations.")
    parser.add_argument("--annotation", type=Path, help="Explicit annotation JSON path.")
    parser.add_argument("--struct", type=Path, help="Explicit struct JSON path.")
    parser.add_argument("--annotations-dir", type=Path, default=Path("annotations"))
    parser.add_argument("--struct-dir", type=Path, default=Path("struct"))
    parser.add_argument("--library", type=Path, default=Path("knowledge/call_mix_library.json"))
    parser.add_argument("--tolerance", type=float, default=0.08, help="Seconds tolerance for timeline/downbeat checks.")
    args = parser.parse_args()

    annotation_paths = resolve_annotation_paths(args)
    if not annotation_paths:
        print("No annotation files matched.", file=sys.stderr)
        return 2
    if len(annotation_paths) > 1 and args.struct:
        print("--struct can only be used with a single annotation.", file=sys.stderr)
        return 2

    total_errors = 0
    for index, annotation_path in enumerate(annotation_paths):
        if index:
            print("\n" + "=" * 72 + "\n")
        findings, struct_path = validate_one(annotation_path, args)
        print_report(annotation_path, struct_path, findings)
        total_errors += sum(1 for finding in findings if finding.severity == "ERROR")

    return 1 if total_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
