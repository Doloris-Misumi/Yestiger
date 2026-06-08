import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ROLE_TO_CATEGORY = {
    "keepspace": "keepspace",
    "rhythmcall": "rhythmcall",
    "mix": "mix",
    "underground_gei": "underground_gei",
}

FALLBACK_ACTIONS = {
    "keepspace": ["keepspace"],
    "rhythmcall": ["sing_along", "clap", "fufu_call", "hai_hai", "ppph", "name_call", "vocal_chant", "oi_oi"],
    "mix": ["standard_mix", "myohon_activation", "myhontousuke", "tiger_fire_activation", "japanese_mix", "ainu_mix", "ietora"],
    "underground_gei": ["long_zhi_mao", "lei_she", "tian_zhao", "dian_bo_she", "zi_he_she", "cun_zheng"],
}

RISK_PENALTY = {
    "low": 0.0,
    "medium": 0.04,
    "high": 0.20,
}

RISK_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def action_family(action_id: str) -> str:
    if "ietora" in action_id or "ie_tiger" in action_id:
        return "ietora"
    if "standard_mix" in action_id:
        return "standard_mix"
    if "japanese_mix" in action_id:
        return "japanese_mix"
    if "ainu_mix" in action_id:
        return "ainu_mix"
    if "activation" in action_id or "myohon" in action_id or "tiger_fire" in action_id:
        return "activation"
    if "mix" in action_id:
        return action_id
    return action_id


def is_repeatable_planned_action(action_id: str, planned_bars: int, library: Dict[str, Dict[str, Any]]) -> bool:
    action = library.get(action_id) or {}
    if str(action.get("category") or "") != "mix":
        return True
    family = action_family(action_id)
    if family in {"ietora", "activation"}:
        return True
    return int(planned_bars) <= 2


def is_nonrepeatable_planned_action(action_id: str, planned_bars: int, library: Dict[str, Dict[str, Any]]) -> bool:
    return not is_repeatable_planned_action(action_id, planned_bars, library)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def coarse_music_label(label: Any) -> str:
    text = str(label or "unknown")
    if text in {"verse", "pre_chorus", "pre_chorus_build", "chant"}:
        return "verse"
    if text in {"chorus", "post_chorus"}:
        return "chorus"
    if text in {"inst", "instrumental", "instrumental_break", "interlude"}:
        return "inst"
    if text in {"intro", "solo", "bridge", "outro", "end"}:
        return text
    return "unknown"


def duration_bucket(bars: float) -> str:
    if bars <= 1.5:
        return "very_short"
    if bars <= 3.5:
        return "short"
    if bars <= 6.5:
        return "medium"
    if bars <= 12.5:
        return "long"
    return "very_long"


def position_bucket(start: float, song_end: float) -> str:
    if song_end <= 0:
        return "unknown"
    pos = start / song_end
    if pos < 0.18:
        return "early"
    if pos < 0.55:
        return "middle"
    if pos < 0.82:
        return "late"
    return "ending"


def load_library(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    return {
        str(action["id"]): action
        for action in data.get("actions") or []
        if isinstance(action, dict) and action.get("id")
    }


def action_display(action_id: str, library: Dict[str, Dict[str, Any]]) -> str:
    action = library.get(action_id)
    if not action:
        return action_id
    return str(action.get("display_name") or action_id)


def action_typical_text(action_id: str, library: Dict[str, Dict[str, Any]]) -> str:
    action = library.get(action_id)
    if not action:
        return ""
    return str(action.get("typical_text") or "")


def context_tags(music_label: str, struct_label: str, role: str, bars: float) -> List[str]:
    tags = {music_label, struct_label, coarse_music_label(music_label), coarse_music_label(struct_label), role}
    if music_label == "intro":
        tags.add("intro")
        if bars >= 6:
            tags.add("long_intro")
    if music_label in {"instrumental_break", "inst", "interlude", "solo"} or struct_label in {"inst", "instrumental"}:
        tags.update({"instrumental_break", "high_energy_break"})
    if music_label in {"post_chorus"}:
        tags.add("post_chorus_interlude")
    if music_label in {"pre_chorus", "pre_chorus_build"}:
        tags.update({"pre_chorus", "chorus_entry", "high_tension_gap"})
    if music_label == "chorus":
        tags.update({"chorus", "chorus_entry", "repeated_chorus"})
    if music_label == "verse":
        tags.add("mid_energy_vocal")
    if music_label == "outro":
        tags.add("high_energy_outro")
    if role == "underground_gei":
        tags.update({"high_energy_break", "high_intensity"})
    return sorted(tag for tag in tags if tag and tag != "unknown")


def load_song_records(bars_dir: Path) -> List[Dict[str, Any]]:
    records = []
    for bars_path in sorted(bars_dir.glob("*/*.signal_bars.jsonl")):
        song_id = bars_path.parent.name
        metrics_path = bars_path.parent / f"{song_id}.signal_metrics.json"
        if not metrics_path.exists():
            continue
        metrics = load_json(metrics_path)
        annotation_path = Path(str(metrics["annotation"]))
        rows = list(load_jsonl(bars_path))
        rows.sort(key=lambda row: int(row.get("bar_index", 0)))
        records.append(
            {
                "song_id": song_id,
                "bars_path": bars_path,
                "metrics": metrics,
                "annotation_path": annotation_path,
                "annotation": load_json(annotation_path),
                "rows": rows,
            }
        )
    return records


def weighted_label_from_rows(rows: Sequence[Dict[str, Any]], start: float, end: float, source: str) -> str:
    weights: Counter[str] = Counter()
    for row in rows:
        row_start = safe_float(row.get("start"))
        row_end = safe_float(row.get("end"))
        overlap = overlap_seconds(start, end, row_start, row_end)
        if overlap <= 0:
            continue
        if source == "music":
            label = str(((row.get("target") or {}).get("music_label")) or "unknown")
        else:
            label = str(((row.get("features") or {}).get("allin1_struct_label")) or "unknown")
        weights[label] += overlap
    if not weights:
        return "unknown"
    return weights.most_common(1)[0][0]


def mean_signal(rows: Sequence[Dict[str, Any]], start: float, end: float) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    total = 0.0
    for row in rows:
        row_start = safe_float(row.get("start"))
        row_end = safe_float(row.get("end"))
        overlap = overlap_seconds(start, end, row_start, row_end)
        if overlap <= 0:
            continue
        signal = row.get("signal_features") or {}
        novelty = row.get("novelty") or {}
        for key in ("energy", "onset", "vocal_density_proxy", "beat_stability"):
            sums[key] += safe_float(signal.get(key)) * overlap
        sums["fused_novelty"] += safe_float(novelty.get("fused")) * overlap
        total += overlap
    if total <= 0:
        return {}
    return {key: round(value / total, 4) for key, value in sums.items()}


def song_end_from_rows(rows: Sequence[Dict[str, Any]]) -> float:
    return max((safe_float(row.get("end")) for row in rows), default=0.0)


def bars_in_annotation_span(rows: Sequence[Dict[str, Any]], start: float, end: float) -> float:
    count = 0.0
    for row in rows:
        row_start = safe_float(row.get("start"))
        row_end = safe_float(row.get("end"))
        if overlap_seconds(start, end, row_start, row_end) > 0.1:
            count += 1.0
    if count > 0:
        return count
    full_durations = [safe_float((row.get("features") or {}).get("duration")) for row in rows if row.get("bar_kind") == "full_bar"]
    median_bar = sorted(full_durations)[len(full_durations) // 2] if full_durations else 2.5
    return max(0.1, (end - start) / max(0.1, median_bar))


def best_music_from_annotation(annotation: Dict[str, Any], start: float, end: float) -> str:
    weights: Counter[str] = Counter()
    for segment in annotation.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        label = str(segment.get("music_label") or "unknown")
        overlap = overlap_seconds(start, end, safe_float(segment.get("start")), safe_float(segment.get("end")))
        if overlap > 0:
            weights[label] += overlap
    if not weights:
        return "unknown"
    return weights.most_common(1)[0][0]


def build_training_examples(records: Sequence[Dict[str, Any]], held_out_song: str) -> List[Dict[str, Any]]:
    examples = []
    for record in records:
        if record["song_id"] == held_out_song:
            continue
        rows = record["rows"]
        annotation = record["annotation"]
        song_end = song_end_from_rows(rows)
        for index, span in enumerate(annotation.get("call_spans") or []):
            if not isinstance(span, dict):
                continue
            actions = [str(action_id) for action_id in span.get("recommended_actions") or [] if str(action_id)]
            role = str(span.get("call_role") or "keepspace")
            if role == "keepspace" or not actions:
                continue
            start = safe_float(span.get("start"))
            end = safe_float(span.get("end"))
            if end <= start:
                continue
            music_label = best_music_from_annotation(annotation, start, end)
            struct_label = weighted_label_from_rows(rows, start, end, "struct")
            bars = bars_in_annotation_span(rows, start, end)
            for action_id in actions:
                examples.append(
                    {
                        "song_id": record["song_id"],
                        "span_index": index,
                        "action_id": action_id,
                        "call_role": role,
                        "music_label": music_label,
                        "coarse_music_label": coarse_music_label(music_label),
                        "struct_label": struct_label,
                        "duration_bucket": duration_bucket(bars),
                        "position_bucket": position_bucket(start, song_end),
                        "bars": bars,
                    }
                )
    return examples


def duration_fit_score(action: Dict[str, Any], bars: float) -> Tuple[float, List[str]]:
    notes = []
    requirements = action.get("requires") or {}
    allowed = requirements.get("allowed_bars")
    min_bars = requirements.get("min_bars")
    max_bars = requirements.get("max_bars")

    if allowed:
        nearest = min(abs(float(item) - bars) for item in allowed)
        score = max(0.25, 1.0 - nearest / max(1.0, bars))
        if nearest > 1.0:
            notes.append(f"duration differs from allowed bars by {nearest:.1f}")
        return score, notes

    score = 1.0
    if min_bars is not None and bars < float(min_bars):
        score *= max(0.25, bars / max(0.1, float(min_bars)))
        notes.append(f"shorter than preferred minimum {float(min_bars):.1f} bars")
    if max_bars is not None and bars > float(max_bars):
        score *= max(0.45, float(max_bars) / max(0.1, bars))
        notes.append(f"longer than preferred maximum {float(max_bars):.1f} bars")
    return max(0.0, min(1.0, score)), notes


def library_score(action_id: str, role: str, tags: Sequence[str], bars: float, library: Dict[str, Dict[str, Any]]) -> Tuple[float, List[str]]:
    action = library.get(action_id)
    if not action:
        return 0.45, ["seen in training annotations but missing from knowledge library"]

    reasons = []
    expected_category = ROLE_TO_CATEGORY.get(role, role)
    category = str(action.get("category") or "")
    if category == expected_category:
        category_score = 1.0
        reasons.append(f"category matches role `{role}`")
    elif action_id in FALLBACK_ACTIONS.get(role, []):
        category_score = 0.8
        reasons.append(f"fallback action for role `{role}`")
    else:
        category_score = 0.2
        reasons.append(f"category `{category}` is not a direct role match")

    best_context = set(action.get("best_context") or [])
    tag_set = set(tags)
    overlap = sorted(best_context & tag_set)
    context_score = min(1.0, 0.45 + 0.18 * len(overlap)) if overlap else 0.35
    if overlap:
        reasons.append("context match: " + ", ".join(overlap[:4]))
    else:
        reasons.append("no exact knowledge-library context match")

    dur_score, dur_notes = duration_fit_score(action, bars)
    reasons.extend(dur_notes)
    risk = str(action.get("risk") or "medium")
    score = 0.40 * category_score + 0.34 * context_score + 0.26 * dur_score - RISK_PENALTY.get(risk, 0.04)
    return max(0.0, min(1.0, score)), reasons


def prototype_scores(examples: Sequence[Dict[str, Any]], query: Dict[str, Any], strategy: str = "balanced") -> Dict[str, Dict[str, Any]]:
    raw_scores: Dict[str, float] = defaultdict(float)
    per_action_scores: Dict[str, List[float]] = defaultdict(list)
    matched: Dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        if example["call_role"] != query["call_role"]:
            continue
        score = 0.18
        if example["music_label"] == query["music_label"]:
            score += 0.34
            matched[example["action_id"]]["music_label"] += 1
        if example["coarse_music_label"] == query["coarse_music_label"]:
            score += 0.16
            matched[example["action_id"]]["coarse_music_label"] += 1
        if example["struct_label"] == query["struct_label"]:
            score += 0.10
            matched[example["action_id"]]["struct_label"] += 1
        if example["duration_bucket"] == query["duration_bucket"]:
            score += 0.12
            matched[example["action_id"]]["duration_bucket"] += 1
        if example["position_bucket"] == query["position_bucket"]:
            score += 0.07
            matched[example["action_id"]]["position_bucket"] += 1
        distance = abs(float(example["bars"]) - float(query["bars"]))
        score += 0.08 * math.exp(-distance / 6.0)
        raw_scores[example["action_id"]] += score
        per_action_scores[example["action_id"]].append(score)

    if not raw_scores:
        return {}

    if strategy == "frequency":
        normalized_scores = dict(raw_scores)
    else:
        normalized_scores = {}
        for action_id, scores in per_action_scores.items():
            ranked_scores = sorted(scores, reverse=True)
            top_scores = ranked_scores[:3]
            best_score = top_scores[0]
            top_mean = sum(top_scores) / len(top_scores)
            full_mean = sum(scores) / len(scores)
            evidence_bonus = 0.03 * min(1.0, math.log1p(len(scores)) / math.log(8.0))
            normalized_scores[action_id] = 0.64 * best_score + 0.26 * top_mean + 0.10 * full_mean + evidence_bonus

    max_score = max(normalized_scores.values())
    output = {}
    for action_id, score in normalized_scores.items():
        output[action_id] = {
            "score": score / max_score if max_score > 0 else 0.0,
            "evidence_count": sum(1 for example in examples if example["call_role"] == query["call_role"] and example["action_id"] == action_id),
            "matched_fields": dict(matched[action_id]),
        }
    return output


def collect_candidate_action_ids(role: str, prototype: Dict[str, Dict[str, Any]], library: Dict[str, Dict[str, Any]]) -> List[str]:
    ids = set(prototype)
    ids.update(FALLBACK_ACTIONS.get(role, []))
    expected_category = ROLE_TO_CATEGORY.get(role)
    for action_id, action in library.items():
        if str(action.get("category") or "") == expected_category:
            ids.add(action_id)
    return sorted(ids)


def rank_actions(query: Dict[str, Any], examples: Sequence[Dict[str, Any]], library: Dict[str, Dict[str, Any]], strategy: str = "balanced") -> List[Dict[str, Any]]:
    role = query["call_role"]
    if role == "keepspace":
        return [
            {
                "action_id": "keepspace",
                "display_name": action_display("keepspace", library),
                "score": 1.0,
                "prototype_score": 1.0,
                "library_score": 1.0,
                "risk": "low",
                "category": "keepspace",
                "typical_text": "",
                "selection_reasons": ["role is keepspace"],
            }
        ]

    prototype = prototype_scores(examples, query, strategy=strategy)
    candidates = []
    for action_id in collect_candidate_action_ids(role, prototype, library):
        action = library.get(action_id)
        proto = prototype.get(action_id, {"score": 0.0, "evidence_count": 0, "matched_fields": {}})
        lib_score, reasons = library_score(action_id, role, query["context_tags"], query["bars"], library)
        if proto["score"] <= 0 and lib_score < 0.45 and action_id not in FALLBACK_ACTIONS.get(role, []):
            continue
        if strategy == "frequency":
            score = 0.62 * float(proto["score"]) + 0.38 * lib_score
        else:
            score = 0.50 * float(proto["score"]) + 0.50 * lib_score
        risk = str((action or {}).get("risk") or "medium")
        matched_fields = proto["matched_fields"]
        if risk == "high" and not any(field in matched_fields for field in ("music_label", "coarse_music_label", "struct_label")):
            score *= 0.70 if strategy == "frequency" else 0.60
            reasons.append("high-risk action down-weighted because only weak context fields matched")
        if risk == "high" and query["bars"] < 4:
            score *= 0.75 if strategy == "frequency" else 0.62
            reasons.append("high-risk action down-weighted for a short slot")
        if strategy in {"balanced", "barfit"} and role == "mix" and risk == "high":
            score -= 0.12
            reasons.append("balanced strategy down-weights high-risk mix actions")
            if action_family(action_id) == "ietora":
                score -= 0.08
                reasons.append("balanced strategy limits repeated ietora-style selections")
        if proto["evidence_count"]:
            reasons.insert(0, f"seen {proto['evidence_count']} times in training songs for role `{role}`")
        else:
            reasons.insert(0, "knowledge-library fallback, not seen in matching training role")
        candidates.append(
            {
                "action_id": action_id,
                "display_name": action_display(action_id, library),
                "score": round(max(0.0, min(1.0, score)), 4),
                "prototype_score": round(float(proto["score"]), 4),
                "library_score": round(float(lib_score), 4),
                "evidence_count": int(proto["evidence_count"]),
                "matched_fields": matched_fields,
                "category": str((action or {}).get("category") or ROLE_TO_CATEGORY.get(role, role)),
                "risk": risk,
                "typical_text": action_typical_text(action_id, library),
                "selection_reasons": reasons[:6],
            }
        )

    candidates.sort(
        key=lambda item: (
            float(item["score"]),
            int(item.get("evidence_count") or 0),
            -RISK_RANK.get(str(item.get("risk") or "medium"), 2),
        ),
        reverse=True,
    )
    return candidates


def preferred_bars(action_id: str, library: Dict[str, Dict[str, Any]]) -> float:
    action = library.get(action_id) or {}
    duration = action.get("duration") or {}
    if duration.get("preferred_bars") is not None:
        return float(duration["preferred_bars"])
    requirements = action.get("requires") or {}
    if requirements.get("min_bars") is not None:
        return max(1.0, float(requirements["min_bars"]))
    return 4.0


def action_bar_options(action_id: str, library: Dict[str, Dict[str, Any]], available_bars: int) -> List[int]:
    if available_bars <= 0:
        return []
    action = library.get(action_id) or {}
    requirements = action.get("requires") or {}
    duration = action.get("duration") or {}
    preferred = max(1, int(round(preferred_bars(action_id, library))))

    allowed = requirements.get("allowed_bars")
    if allowed:
        options = sorted({int(round(float(item))) for item in allowed if 0 < int(round(float(item))) <= available_bars})
        return sorted(options, key=lambda value: (abs(value - preferred), value))

    min_bars = int(math.ceil(safe_float(requirements.get("min_bars"), preferred)))
    max_bars = int(math.floor(safe_float(requirements.get("max_bars"), preferred)))
    min_bars = max(1, min_bars)
    max_bars = max(min_bars, max_bars)
    strict = bool(duration.get("strict_bars")) and not bool(duration.get("can_extend"))
    bar_multiple = requirements.get("bar_multiple")

    if strict:
        options = [preferred] if min_bars <= preferred <= max_bars else [min_bars]
    else:
        options = list(range(min_bars, min(max_bars, available_bars) + 1))
        if preferred <= available_bars and preferred not in options and min_bars <= preferred <= max_bars:
            options.append(preferred)

    if bar_multiple:
        multiple = max(1, int(round(safe_float(bar_multiple, 1.0))))
        options = [value for value in options if value % multiple == 0]
    options = sorted({value for value in options if 0 < value <= available_bars})
    return sorted(options, key=lambda value: (abs(value - preferred), value))


def duration_option_score(action_id: str, planned_bars: int, library: Dict[str, Dict[str, Any]]) -> float:
    action = library.get(action_id) or {}
    requirements = action.get("requires") or {}
    allowed = requirements.get("allowed_bars")
    if allowed:
        allowed_values = {int(round(float(item))) for item in allowed}
        if int(planned_bars) in allowed_values:
            return 1.0
    preferred = max(1.0, preferred_bars(action_id, library))
    return max(0.0, 1.0 - abs(float(planned_bars) - preferred) / max(preferred, float(planned_bars), 1.0))


def clone_with_planned_bars(candidate: Dict[str, Any], planned_bars: int, library: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    item = dict(candidate)
    item["planned_bars"] = int(planned_bars)
    item["duration_fit"] = round(duration_option_score(str(item["action_id"]), int(planned_bars), library), 4)
    return item


def select_barfit_actions(
    role: str,
    bars: float,
    ranked: Sequence[Dict[str, Any]],
    library: Dict[str, Dict[str, Any]],
    used_nonrepeatable_mix_actions: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if role == "keepspace" or not ranked:
        return []
    target_bars = max(1, int(round(bars)))

    used_nonrepeatable_mix_actions = used_nonrepeatable_mix_actions or set()
    candidates = [item for item in ranked[:30] if float(item.get("score") or 0.0) >= 0.20]
    if not candidates:
        return []

    max_actions = min(8, max(1, int(math.ceil(target_bars / 4.0)) + 1))
    states: Dict[Tuple[int, int], Tuple[float, List[Dict[str, Any]], Tuple[str, ...]]] = {
        (0, 0): (0.0, [], tuple())
    }
    for rank, candidate in enumerate(candidates):
        action_id = str(candidate["action_id"])
        options = action_bar_options(action_id, library, target_bars)
        if role == "mix" and action_id in used_nonrepeatable_mix_actions:
            options = [
                option
                for option in options
                if is_repeatable_planned_action(action_id, option, library)
            ]
        if not options:
            continue
        family = action_family(action_id)
        current_states = list(states.items())
        for (used_bars, used_actions), (state_score, items, families) in current_states:
            if used_actions >= max_actions:
                continue
            if any(item["action_id"] == action_id for item in items):
                continue
            for option in options:
                new_used = used_bars + option
                if new_used > target_bars:
                    continue
                family_penalty = 0.08 if family in families else 0.0
                risk_penalty = 0.05 if str(candidate.get("risk") or "medium") == "high" else 0.0
                rank_penalty = 0.01 * rank
                option_score = (
                    float(candidate["score"])
                    + 0.18 * duration_option_score(action_id, option, library)
                    - family_penalty
                    - risk_penalty
                    - rank_penalty
                )
                new_score = state_score + option_score
                key = (new_used, used_actions + 1)
                previous = states.get(key)
                if previous is None or new_score > previous[0]:
                    states[key] = (
                        new_score,
                        items + [clone_with_planned_bars(candidate, option, library)],
                        families + (family,),
                    )

    scored_states = []
    for (used_bars, used_actions), (state_score, items, _families) in states.items():
        if not items:
            continue
        fill_ratio = used_bars / max(1, target_bars)
        action_penalty = 0.015 * max(0, used_actions - 1)
        scored_states.append((fill_ratio, state_score / max(1, used_actions) - action_penalty, used_bars, items))
    if not scored_states:
        return []

    scored_states.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return scored_states[0][3]


def select_plan_actions(
    role: str,
    bars: float,
    ranked: Sequence[Dict[str, Any]],
    library: Dict[str, Dict[str, Any]],
    strategy: str = "balanced",
    used_nonrepeatable_mix_actions: Optional[set] = None,
) -> List[Dict[str, Any]]:
    if role == "keepspace":
        return []
    if not ranked:
        return []
    if strategy == "barfit":
        return select_barfit_actions(role, bars, ranked, library, used_nonrepeatable_mix_actions=used_nonrepeatable_mix_actions)

    if role == "rhythmcall":
        count = 2 if bars >= 12 else 1
    elif role == "mix":
        count = 3 if bars >= 16 else 2 if bars >= 8 else 1
    else:
        count = 2 if bars >= 8 else 1

    selected = []
    seen = set()
    seen_families = set()
    if strategy in {"balanced", "barfit"} and role == "mix":
        ordered = sorted(
            ranked,
            key=lambda item: (
                float(item["score"])
                - (0.10 if str(item.get("risk") or "medium") == "high" else 0.0)
                - (0.06 if action_family(str(item.get("action_id") or "")) == "ietora" else 0.0),
                -RISK_RANK.get(str(item.get("risk") or "medium"), 2),
                int(item.get("evidence_count") or 0),
            ),
            reverse=True,
        )
    else:
        ordered = list(ranked)

    for candidate in ordered:
        action_id = candidate["action_id"]
        if action_id in seen:
            continue
        if strategy in {"balanced", "barfit"} and role == "mix":
            family = action_family(action_id)
            if family in seen_families and len(selected) + 1 < count:
                continue
            if (
                not selected
                and str(candidate.get("risk") or "medium") == "high"
                and bars < 4
                and any(str(item.get("risk") or "medium") != "high" and float(item["score"]) >= float(candidate["score"]) - 0.20 for item in ranked)
            ):
                continue
        if candidate["score"] < 0.25 and selected:
            continue
        selected.append(candidate)
        seen.add(action_id)
        if strategy in {"balanced", "barfit"} and role == "mix":
            seen_families.add(action_family(action_id))
        if len(selected) >= count:
            break
    if not selected:
        selected.append(ranked[0])
    return selected


def rows_overlapping_span(rows: Sequence[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    return [
        row
        for row in rows
        if overlap_seconds(start, end, safe_float(row.get("start")), safe_float(row.get("end"))) > 0.1
    ]


def bar_slice_time(
    span_start: float,
    span_end: float,
    rows: Sequence[Dict[str, Any]],
    offset_bars: int,
    count_bars: int,
    total_bars: int,
) -> Tuple[float, float]:
    if rows and offset_bars < len(rows):
        start_index = max(0, min(offset_bars, len(rows) - 1))
        end_index = max(start_index, min(offset_bars + count_bars - 1, len(rows) - 1))
        return (
            max(span_start, safe_float(rows[start_index].get("start"))),
            min(span_end, safe_float(rows[end_index].get("end"))),
        )
    seconds_per_bar = (span_end - span_start) / max(1, total_bars)
    sub_start = span_start + seconds_per_bar * offset_bars
    sub_end = span_start + seconds_per_bar * (offset_bars + count_bars)
    return max(span_start, sub_start), min(span_end, sub_end)


def make_action_plan(
    span: Dict[str, Any],
    selected: Sequence[Dict[str, Any]],
    library: Dict[str, Dict[str, Any]],
    rows: Optional[Sequence[Dict[str, Any]]] = None,
    strategy: str = "balanced",
) -> List[Dict[str, Any]]:
    start = safe_float(span.get("start"))
    end = safe_float(span.get("end"))
    role = str(span.get("call_role") or "keepspace")
    bars = safe_float(span.get("bars"), max(0.1, len(selected)))
    target_bars = max(1, int(round(bars)))
    if not selected:
        if strategy == "barfit" and role != "keepspace":
            return [
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "time": f"{fmt_time(start)}-{fmt_time(end)}",
                    "action_id": "keepspace",
                    "display_name": "Keep Space / Too Short for Action",
                    "typical_text": "",
                    "risk": "low",
                    "bar_start_offset": 0,
                    "bar_count": target_bars,
                    "duration_fit": 1.0,
                    "mode": "barfit_gap",
                }
            ]
        return []

    if strategy == "barfit":
        span_rows = rows_overlapping_span(rows or [], start, end)
        cursor_bars = 0
        plan = []
        for action in selected:
            planned_bars = max(1, int(action.get("planned_bars") or round(preferred_bars(str(action["action_id"]), library))))
            planned_bars = min(planned_bars, max(0, target_bars - cursor_bars))
            if planned_bars <= 0:
                continue
            sub_start, sub_end = bar_slice_time(start, end, span_rows, cursor_bars, planned_bars, target_bars)
            plan.append(
                {
                    "start": round(sub_start, 3),
                    "end": round(sub_end, 3),
                    "time": f"{fmt_time(sub_start)}-{fmt_time(sub_end)}",
                    "action_id": action["action_id"],
                    "display_name": action["display_name"],
                    "typical_text": action.get("typical_text", ""),
                    "risk": action.get("risk"),
                    "bar_start_offset": cursor_bars,
                    "bar_count": planned_bars,
                    "duration_fit": action.get("duration_fit", duration_option_score(str(action["action_id"]), planned_bars, library)),
                    "mode": "barfit_action",
                }
            )
            cursor_bars += planned_bars
        if cursor_bars < target_bars:
            sub_start, sub_end = bar_slice_time(start, end, span_rows, cursor_bars, target_bars - cursor_bars, target_bars)
            plan.append(
                {
                    "start": round(sub_start, 3),
                    "end": round(sub_end, 3),
                    "time": f"{fmt_time(sub_start)}-{fmt_time(sub_end)}",
                    "action_id": "keepspace",
                    "display_name": "Keep Space / Unassigned Gap",
                    "typical_text": "",
                    "risk": "low",
                    "bar_start_offset": cursor_bars,
                    "bar_count": target_bars - cursor_bars,
                    "duration_fit": 1.0,
                    "mode": "barfit_gap",
                }
            )
        return plan

    if role == "rhythmcall" and len(selected) == 1:
        action = selected[0]
        return [
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "time": f"{fmt_time(start)}-{fmt_time(end)}",
                "action_id": action["action_id"],
                "display_name": action["display_name"],
                "typical_text": action.get("typical_text", ""),
                "risk": action.get("risk"),
                "bar_count": target_bars,
                "duration_fit": action.get("duration_fit", 1.0),
                "mode": "repeat_or_follow_phrase",
            }
        ]

    weights = [max(1.0, preferred_bars(item["action_id"], library)) for item in selected]
    total = sum(weights)
    cursor = start
    plan = []
    for index, (action, weight) in enumerate(zip(selected, weights)):
        if index == len(selected) - 1:
            sub_end = end
        else:
            sub_end = cursor + (end - start) * (weight / total)
        plan.append(
            {
                "start": round(cursor, 3),
                "end": round(sub_end, 3),
                "time": f"{fmt_time(cursor)}-{fmt_time(sub_end)}",
                "action_id": action["action_id"],
                "display_name": action["display_name"],
                "typical_text": action.get("typical_text", ""),
                "risk": action.get("risk"),
                "mode": "timed_subaction" if len(selected) > 1 else "primary_action",
            }
        )
        cursor = sub_end
    return plan


def enrich_span(
    span: Dict[str, Any],
    record: Dict[str, Any],
    examples: Sequence[Dict[str, Any]],
    library: Dict[str, Dict[str, Any]],
    strategy: str = "balanced",
    used_nonrepeatable_mix_actions: Optional[set] = None,
) -> Dict[str, Any]:
    start = safe_float(span.get("start"))
    end = safe_float(span.get("end"))
    role = str(span.get("call_role") or "keepspace")
    bars = safe_float(span.get("bars"), 0.0)
    rows = record["rows"]
    song_end = song_end_from_rows(rows)
    music_label = weighted_label_from_rows(rows, start, end, "music")
    struct_label = weighted_label_from_rows(rows, start, end, "struct")
    query = {
        "call_role": role,
        "music_label": music_label,
        "coarse_music_label": coarse_music_label(music_label),
        "struct_label": struct_label,
        "duration_bucket": duration_bucket(bars),
        "position_bucket": position_bucket(start, song_end),
        "bars": bars,
        "context_tags": context_tags(music_label, struct_label, role, bars),
    }
    ranked = rank_actions(query, examples, library, strategy=strategy)
    selected = select_plan_actions(
        role,
        bars,
        ranked,
        library,
        strategy=strategy,
        used_nonrepeatable_mix_actions=used_nonrepeatable_mix_actions,
    )
    action_plan = make_action_plan(span, selected, library, rows=rows, strategy=strategy)
    if strategy == "barfit" and used_nonrepeatable_mix_actions is not None:
        for item in selected:
            action_id = str(item.get("action_id") or "")
            planned_bars = int(item.get("planned_bars") or round(preferred_bars(action_id, library)))
            if is_nonrepeatable_planned_action(action_id, planned_bars, library):
                used_nonrepeatable_mix_actions.add(action_id)
    enriched = dict(span)
    enriched.update(
        {
            "music_label_context": music_label,
            "allin1_struct_context": struct_label,
            "context_tags": query["context_tags"],
            "signal_summary": mean_signal(rows, start, end),
            "recommended_actions": [item["action_id"] for item in selected],
            "action_plan": action_plan,
            "action_candidates": ranked[:8],
            "action_selection": {
                "mode": f"loso_annotation_prototype_plus_knowledge_library_{strategy}",
                "held_out_song": record["song_id"],
                "training_action_examples": len(examples),
                "used_nonrepeatable_mix_actions": sorted(used_nonrepeatable_mix_actions or []),
                "note": "Held-out song annotation actions are not used for selection.",
            },
        }
    )
    return enriched


def write_markdown(path: Path, song_id: str, method: str, selector_name: str, spans: Sequence[Dict[str, Any]]) -> None:
    lines = [
        f"# Action-Enriched Signal Callbook: {song_id}",
        "",
        f"Base method: `{method}`",
        "",
        f"Action selector: `{selector_name}`",
        "",
        "| Time | Role | Context | Bars | Actions | Typical Text / Notes |",
        "|---:|---|---|---:|---|---|",
    ]
    for span in spans:
        context = f"{span.get('music_label_context', '-')}/{span.get('allin1_struct_context', '-')}"
        action_lines = []
        text_lines = []
        for action in span.get("action_plan") or []:
            risk = action.get("risk")
            risk_suffix = f" [{risk}]" if risk in {"medium", "high"} else ""
            bar_suffix = f" ({action['bar_count']} bars)" if action.get("bar_count") is not None else ""
            fit = action.get("duration_fit")
            fit_suffix = f" fit={float(fit):.2f}" if fit is not None and float(fit) < 0.999 else ""
            action_lines.append(f"{action['time']} {action['display_name']}{risk_suffix}{bar_suffix}{fit_suffix}")
            if action.get("typical_text"):
                text_lines.append(f"`{action['typical_text']}`")
        if not action_lines:
            action_lines.append("Keep Space")
        top_candidates = span.get("action_candidates") or []
        notes = []
        if top_candidates:
            first = top_candidates[0]
            notes.append(f"top score={first.get('score'):.3f}")
            reasons = first.get("selection_reasons") or []
            if reasons:
                notes.append(reasons[0])
        text = "<br>".join(text_lines + notes) if text_lines or notes else "-"
        lines.append(
            f"| {fmt_time(safe_float(span.get('start')))}-{fmt_time(safe_float(span.get('end')))} | "
            f"{span.get('call_role')} | {context} | {span.get('bars')} | "
            f"{'<br>'.join(action_lines)} | {text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enrich_callbook(
    call_spans_path: Path,
    records: Sequence[Dict[str, Any]],
    library: Dict[str, Dict[str, Any]],
    out_dir: Optional[Path],
    strategy: str = "balanced",
    output_tag: str = "balanced_action",
) -> Tuple[Path, Path]:
    payload = load_json(call_spans_path)
    song_id = str(payload.get("song_id") or call_spans_path.parent.name)
    method = str(payload.get("method") or "unknown_method")
    record_by_song = {record["song_id"]: record for record in records}
    if song_id not in record_by_song:
        raise SystemExit(f"No signal bars record found for song_id={song_id}")
    record = record_by_song[song_id]
    examples = build_training_examples(records, held_out_song=song_id)
    selector_name = f"loso_annotation_prototype_plus_knowledge_library_{strategy}"
    used_nonrepeatable_mix_actions = set()
    enriched_spans = [
        enrich_span(
            span,
            record,
            examples,
            library,
            strategy=strategy,
            used_nonrepeatable_mix_actions=used_nonrepeatable_mix_actions,
        )
        for span in payload.get("call_spans") or []
    ]
    result = {
        "song_id": song_id,
        "base_method": method,
        "source_call_spans": str(call_spans_path),
        "action_selector": {
            "name": selector_name,
            "strategy": strategy,
            "held_out_song": song_id,
            "training_action_examples": len(examples),
            "nonrepeatable_mix_rule": "For barfit, mix actions planned for 3 or more bars are used at most once per song; short mix actions and ietora/activation-style triggers may repeat.",
            "uses_tiny_action_ranker": False,
            "leakage_note": "The held-out song's recommended_actions are not used for action selection.",
        },
        "call_spans": enriched_spans,
    }
    target_dir = out_dir or call_spans_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = call_spans_path.name.replace(".call_spans.json", "")
    json_path = target_dir / f"{stem}.{output_tag}_call_spans.json"
    md_path = target_dir / f"{stem}.{output_tag}_callbook.md"
    write_json(json_path, result)
    write_markdown(md_path, song_id, method, selector_name, enriched_spans)
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Add concrete recommended actions to merged role-level signal callbooks.")
    parser.add_argument("--call-spans", type=Path, required=True)
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--library", type=Path, default=Path("knowledge/call_mix_library.json"))
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--strategy", choices=["frequency", "balanced", "barfit"], default="balanced")
    parser.add_argument("--output-tag", default=None)
    args = parser.parse_args()

    records = load_song_records(args.bars_dir)
    library = load_library(args.library)
    output_tag = args.output_tag or {"frequency": "action", "balanced": "balanced_action", "barfit": "barfit_action"}[args.strategy]
    json_path, md_path = enrich_callbook(args.call_spans, records, library, args.out_dir, strategy=args.strategy, output_tag=output_tag)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
