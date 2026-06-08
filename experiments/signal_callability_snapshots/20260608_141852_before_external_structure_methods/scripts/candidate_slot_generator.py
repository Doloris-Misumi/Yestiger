import argparse
import itertools
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Tuple


RISK_PENALTY = {
    "low": 0.0,
    "medium": 0.04,
    "high": 0.10,
}
RISK_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path_text: str, base_dir: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return base_dir / path


def estimate_bar_seconds(struct_data: Optional[Dict[str, Any]]) -> float:
    if struct_data:
        downbeats = struct_data.get("downbeats") or []
        if len(downbeats) >= 2:
            diffs = [b - a for a, b in zip(downbeats[:-1], downbeats[1:]) if b > a]
            if diffs:
                return float(median(diffs))
        bpm = struct_data.get("bpm")
        if bpm:
            return 60.0 / float(bpm) * 4.0
    return 2.54


def count_bars(start: float, end: float, struct_data: Optional[Dict[str, Any]], bar_seconds: float) -> float:
    downbeats = (struct_data or {}).get("downbeats") or []
    if downbeats:
        inner = [t for t in downbeats if start <= t < end]
        if len(inner) >= 2:
            return float(len(inner) - 1)
    return max(0.1, (end - start) / bar_seconds)


def context_tags(segment: Dict[str, Any], index: int, segments: List[Dict[str, Any]]) -> List[str]:
    label = segment["label"]
    parent = segment.get("parent_label", label)
    duration = segment["end"] - segment["start"]
    prev_label = segments[index - 1]["label"] if index > 0 else None
    next_label = segments[index + 1]["label"] if index + 1 < len(segments) else None

    tags = {label, parent}

    if label == "intro":
        tags.add("intro")
        if duration >= 18:
            tags.add("long_intro")
    elif label == "verse_subsection":
        tags.update(["verse", "mid_energy_vocal"])
    elif label == "pre_chorus_build_candidate":
        tags.update(["pre_chorus", "pre_chorus_build", "high_tension_gap"])
    elif label == "pre_chorus_candidate":
        tags.update(["pre_chorus"])
    elif label == "post_chorus_interlude_candidate":
        tags.update(["post_chorus_interlude", "instrumental_break", "reintro"])
    elif label == "chorus":
        tags.add("chorus")
        if prev_label == "chorus" or next_label == "chorus":
            tags.add("repeated_chorus")
    elif label == "inst":
        tags.update(["instrumental_break"])
        if duration >= 24:
            tags.add("extended_instrumental")
    elif label == "solo":
        tags.update(["instrumental_break", "emotional_solo", "member_focus"])
    elif label == "outro":
        tags.update(["outro", "high_energy_outro"])
    elif label == "end":
        tags.add("end")

    return sorted(tags)


def duration_score(bars: float, requires: Dict[str, Any]) -> Tuple[float, List[str], List[str]]:
    reasons: List[str] = []
    cautions: List[str] = []
    min_bars = requires.get("min_bars")
    max_bars = requires.get("max_bars")

    if min_bars is None and max_bars is None:
        return 0.85, reasons, cautions

    if min_bars is not None and bars < float(min_bars):
        ratio = bars / max(float(min_bars), 0.1)
        cautions.append(f"shorter than preferred minimum ({bars:.1f} < {float(min_bars):.1f} bars)")
        return max(0.15, 0.55 * ratio), reasons, cautions

    if max_bars is not None and bars > float(max_bars):
        ratio = float(max_bars) / max(bars, 0.1)
        cautions.append(f"longer than preferred maximum ({bars:.1f} > {float(max_bars):.1f} bars)")
        return max(0.45, 0.85 * ratio), reasons, cautions

    if min_bars is not None:
        reasons.append(f"fits minimum bar length ({bars:.1f} bars)")
    if max_bars is not None:
        reasons.append(f"within maximum bar length ({bars:.1f} bars)")
    return 1.0, reasons, cautions


def score_action(
    action: Dict[str, Any],
    tags: List[str],
    bars: float,
    segment_confidence: float,
) -> Optional[Dict[str, Any]]:
    best_context = set(action.get("best_context") or [])
    tag_set = set(tags)
    overlap = sorted(best_context & tag_set)

    if not overlap:
        return None

    context_score = min(1.0, 0.62 + 0.18 * len(overlap))
    dur_score, dur_reasons, cautions = duration_score(bars, action.get("requires") or {})
    risk = action.get("risk", "medium")
    risk_penalty = RISK_PENALTY.get(risk, 0.04)

    confidence_factor = 0.75 + 0.25 * max(0.0, min(1.0, float(segment_confidence)))
    score = (0.58 * context_score + 0.30 * dur_score + 0.12 * action.get("intensity", 0.5))
    score = score * confidence_factor - risk_penalty
    score = max(0.0, min(1.0, score))

    reasons = [f"context match: {', '.join(overlap)}"] + dur_reasons
    if risk == "high":
        cautions.append("high-risk call; should be style/profile gated")
    elif risk == "medium":
        cautions.append("medium-risk call; verify local live culture")

    return {
        "action_id": action["id"],
        "display_name": action["display_name"],
        "category": action["category"],
        "score": round(score, 4),
        "intensity": action.get("intensity"),
        "risk": risk,
        "min_bars": action.get("requires", {}).get("min_bars"),
        "max_bars": action.get("requires", {}).get("max_bars"),
        "duration": action.get("duration", {}),
        "typical_text": action.get("typical_text", ""),
        "matched_context": overlap,
        "reasons": reasons,
        "cautions": cautions,
    }


def risk_level(risks: List[str]) -> str:
    if not risks:
        return "low"
    return max(risks, key=lambda risk: RISK_RANK.get(risk, 2))


def preferred_action_bars(candidate: Dict[str, Any], action: Dict[str, Any]) -> float:
    duration = action.get("duration") or candidate.get("duration") or {}
    preferred_bars = duration.get("preferred_bars")
    if preferred_bars is not None:
        return float(preferred_bars)

    action_id = action["id"]
    category = action.get("category")
    requires = action.get("requires") or {}
    min_bars = float(requires.get("min_bars") or candidate.get("min_bars") or 1.0)
    max_bars = float(requires.get("max_bars") or candidate.get("max_bars") or min_bars)

    if action_id == "kaho_sanren_mix":
        return max(8.0, min(min_bars, max_bars))
    if category == "underground_gei":
        return max(8.0, min_bars)
    if category == "mix":
        return max(4.0, min_bars)
    if category == "rhythmcall":
        return max(2.0, min(4.0, min_bars))
    return max(1.0, min_bars)


def can_use_in_chain(candidate: Dict[str, Any], action: Dict[str, Any]) -> bool:
    category = action.get("category")
    if category in {"mix", "underground_gei"}:
        return True
    if category == "rhythmcall":
        return action["id"] in {"oi_oi", "hai_hai"}
    return False


def estimate_chain_times(
    start: float,
    end: float,
    action_bars: List[float],
) -> List[Tuple[float, float]]:
    total = max(0.1, sum(action_bars))
    duration = end - start
    cursor = start
    spans = []
    for index, bars in enumerate(action_bars):
        if index == len(action_bars) - 1:
            sub_end = end
        else:
            sub_end = cursor + duration * (bars / total)
        spans.append((round(cursor, 3), round(sub_end, 3)))
        cursor = sub_end
    return spans


def generate_chain_candidates(
    candidates: List[Dict[str, Any]],
    action_by_id: Dict[str, Dict[str, Any]],
    tags: List[str],
    bars: float,
    start: float,
    end: float,
) -> List[Dict[str, Any]]:
    chain_context = {"extended_instrumental", "instrumental_break", "long_intro", "post_chorus_interlude"}
    if bars < 7.0 or not (set(tags) & chain_context):
        return []

    eligible = []
    for candidate in candidates:
        action = action_by_id[candidate["action_id"]]
        if not can_use_in_chain(candidate, action):
            continue
        eligible.append((candidate, action, preferred_action_bars(candidate, action)))

    if len(eligible) < 2:
        return []

    if bars >= 13:
        max_actions = 4
    elif bars >= 9:
        max_actions = 3
    else:
        max_actions = 2

    chains = []
    for length in range(2, max_actions + 1):
        for sequence in itertools.permutations(eligible, length):
            action_ids = [action["id"] for _, action, _ in sequence]
            if len(set(action_ids)) != len(action_ids):
                continue

            action_bars = [preferred for _, _, preferred in sequence]
            total_bars = sum(action_bars)
            if total_bars > bars + 1.0:
                continue
            if total_bars < bars * 0.55:
                continue

            avg_score = sum(candidate["score"] for candidate, _, _ in sequence) / length
            coverage_score = 1.0 - abs(bars - total_bars) / max(bars, 0.1)
            high_risk_count = sum(1 for candidate, _, _ in sequence if candidate.get("risk") == "high")
            medium_risk_count = sum(1 for candidate, _, _ in sequence if candidate.get("risk") == "medium")
            risk_penalty = high_risk_count * 0.08 + medium_risk_count * 0.025
            length_bonus = 0.04 if length >= 3 and bars >= 9 else 0.0
            score = max(0.0, min(1.0, avg_score * 0.62 + coverage_score * 0.34 + length_bonus - risk_penalty))

            spans = estimate_chain_times(start, end, action_bars)
            actions = []
            for (candidate, action, preferred), (sub_start, sub_end) in zip(sequence, spans):
                actions.append({
                    "action_id": action["id"],
                    "display_name": action["display_name"],
                    "category": action["category"],
                    "risk": candidate["risk"],
                    "preferred_bars": preferred,
                    "time": f"{fmt_time(sub_start)} - {fmt_time(sub_end)}",
                    "start": sub_start,
                    "end": sub_end,
                    "text": candidate.get("typical_text", ""),
                })

            risks = [candidate.get("risk", "medium") for candidate, _, _ in sequence]
            cautions = []
            if high_risk_count:
                cautions.append("contains high-risk action; profile or live culture should allow it")
            if total_bars > bars:
                cautions.append(f"tight chain ({total_bars:.1f} action bars into {bars:.1f} slot bars)")

            chains.append({
                "chain_id": "chain_" + "_".join(action_ids),
                "display_name": " -> ".join(action["display_name"] for _, action, _ in sequence),
                "category": "chain",
                "score": round(score, 4),
                "risk": risk_level(risks),
                "action_count": length,
                "total_preferred_bars": round(total_bars, 2),
                "slot_bars": round(bars, 2),
                "coverage": round(total_bars / max(bars, 0.1), 3),
                "actions": actions,
                "reasons": [
                    f"dynamic chain for {', '.join(sorted(set(tags) & chain_context))}",
                    f"packs {length} atomic actions into {bars:.1f} bars",
                ],
                "cautions": cautions,
            })

    chains.sort(key=lambda chain: chain["score"], reverse=True)
    deduped = []
    seen = set()
    for chain in chains:
        action_tuple = tuple(action["action_id"] for action in chain["actions"])
        if action_tuple in seen:
            continue
        seen.add(action_tuple)
        deduped.append(chain)
        if len(deduped) >= 8:
            break
    return deduped


def generate_slots(secondary: Dict[str, Any], library: Dict[str, Any], struct_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    segments = secondary["refined_segments"]
    actions = library["actions"]
    action_by_id = {action["id"]: action for action in actions}
    bar_seconds = estimate_bar_seconds(struct_data)

    slots = []
    for index, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        tags = context_tags(segment, index, segments)
        bars = count_bars(start, end, struct_data, bar_seconds)
        seg_conf = float(segment.get("confidence", 0.5))

        candidates = []
        for action in actions:
            candidate = score_action(action, tags, bars, seg_conf)
            if candidate:
                candidates.append(candidate)

        candidates.sort(key=lambda x: x["score"], reverse=True)
        chain_candidates = generate_chain_candidates(
            candidates,
            action_by_id,
            tags,
            bars,
            start,
            end,
        )
        slots.append({
            "start": start,
            "end": end,
            "time": f"{fmt_time(start)} - {fmt_time(end)}",
            "duration_seconds": round(end - start, 3),
            "estimated_bars": round(bars, 2),
            "source_label": segment["label"],
            "parent_label": segment.get("parent_label", segment["label"]),
            "segment_confidence": round(seg_conf, 4),
            "context_tags": tags,
            "source_reason": segment.get("reason", ""),
            "candidates": candidates,
            "chain_candidates": chain_candidates,
        })

    return {
        "source_secondary": secondary.get("source_struct"),
        "source_audio": secondary.get("source_audio"),
        "library_version": library.get("version"),
        "bar_seconds_estimate": round(bar_seconds, 4),
        "slots": slots,
    }


def write_markdown(path: Path, result: Dict[str, Any], top_n: int) -> None:
    lines = [
        "# Candidate Slot Report",
        "",
        f"Bar seconds estimate: `{result['bar_seconds_estimate']}`",
        "",
        "| Time | Label | Bars | Context Tags | Top Candidates |",
        "|---:|---|---:|---|---|",
    ]
    for slot in result["slots"]:
        top = slot["candidates"][:top_n]
        top_chains = slot.get("chain_candidates", [])[:2]
        if top:
            top_text = "<br>".join(
                f"{c['display_name']} ({c['score']:.2f}, {c['risk']})"
                for c in top
            )
        else:
            top_text = "No candidate"
        if top_chains:
            chain_text = "<br>".join(
                f"Chain: {chain['display_name']} ({chain['score']:.2f}, {chain['risk']})"
                for chain in top_chains
            )
            top_text = f"{top_text}<br>{chain_text}"
        lines.append(
            f"| {slot['time']} | {slot['source_label']} | {slot['estimated_bars']:.1f} | "
            f"{', '.join(slot['context_tags'])} | {top_text} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--secondary", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--struct", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("ouen_analysis"))
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    secondary = load_json(args.secondary)
    library = load_json(args.library)
    struct_path = args.struct
    if struct_path is None and secondary.get("source_struct"):
        struct_path = resolve_path(secondary["source_struct"], args.secondary.parent.parent)

    struct_data = load_json(struct_path) if struct_path and struct_path.exists() else None

    result = generate_slots(secondary, library, struct_data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.secondary.name.replace(".secondary.json", "")
    json_path = args.out_dir / f"{stem}.candidates.json"
    md_path = args.out_dir / f"{stem}.candidates.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(md_path, result, args.top_n)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
