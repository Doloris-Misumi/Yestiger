import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


POLICY = {
    "intro": ["standard_mix", "clap"],
    "verse_subsection": ["ppph", "clap"],
    "pre_chorus_build_candidate": ["ietora", "ppph", "clap"],
    "pre_chorus_candidate": ["ietora", "ppph", "clap"],
    "post_chorus_interlude_candidate": ["tiger_fire_activation", "standard_mix", "japanese_mix", "clap"],
    "chorus": ["fuwa_fuwa", "hai_hai", "clap"],
    "inst": ["standard_mix", "japanese_mix", "ainu_mix", "kaho_sanren_mix", "gachikoi_koujou", "oi_oi"],
    "solo": ["kecha", "name_call", "standard_mix", "clap"],
    "outro": ["oi_oi", "clap"],
    "end": [],
}


HIGH_RISK_IDS = {"ietora", "gachikoi_koujou", "kaho_sanren_mix"}


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_by_id(slot: Dict[str, Any], action_id: str) -> Optional[Dict[str, Any]]:
    for candidate in slot.get("candidates", []):
        if candidate["action_id"] == action_id:
            return candidate
    return None


def choose_for_slot(slot: Dict[str, Any]) -> Dict[str, Any]:
    label = slot["source_label"]
    candidates = slot.get("candidates", [])
    action = None
    selection_reason = ""

    for action_id in POLICY.get(label, []):
        found = candidate_by_id(slot, action_id)
        if found:
            action = found
            selection_reason = f"Selected by policy preference for {label}: {action_id}."
            break

    if action is None and candidates:
        action = candidates[0]
        selection_reason = "Selected top-scoring candidate."

    if action is None:
        return {
            "start": slot["start"],
            "end": slot["end"],
            "time": slot["time"],
            "slot_label": label,
            "action_id": "keep_space",
            "display_name": "Keep Space",
            "category": "arrangement",
            "intensity": 0.0,
            "risk": "low",
            "confidence": 0.5,
            "alternatives": [],
            "notes": ["No suitable call/MIX candidate; keep this section open."],
        }

    alternatives = [
        {
            "action_id": c["action_id"],
            "display_name": c["display_name"],
            "score": c["score"],
            "risk": c["risk"],
        }
        for c in candidates
        if c["action_id"] != action["action_id"]
    ][:4]

    notes = [selection_reason]
    notes.extend(action.get("cautions") or [])

    if action["action_id"] in HIGH_RISK_IDS:
        notes.append("High-risk action: keep as opt-in until a live-style profile allows it.")

    if slot["estimated_bars"] < 4 and action["category"] == "mix":
        notes.append("Shorter than a full MIX window; use a shortened cue or skip.")

    return {
        "start": slot["start"],
        "end": slot["end"],
        "time": slot["time"],
        "slot_label": label,
        "source_context_tags": slot["context_tags"],
        "estimated_bars": slot["estimated_bars"],
        "action_id": action["action_id"],
        "display_name": action["display_name"],
        "category": action["category"],
        "typical_text": action.get("typical_text", ""),
        "intensity": action.get("intensity"),
        "risk": action.get("risk"),
        "confidence": action.get("score"),
        "alternatives": alternatives,
        "notes": notes,
    }


def arrange(candidates: Dict[str, Any]) -> Dict[str, Any]:
    entries = [choose_for_slot(slot) for slot in candidates["slots"]]

    # Keep the draft less noisy: if two consecutive chorus blocks pick the same
    # action, mark the second as continuation rather than a new instruction.
    previous_action = None
    for entry in entries:
        if entry["slot_label"] == "chorus" and entry["action_id"] == previous_action:
            entry["notes"].append("Continuation of previous chorus call; avoid restarting the pattern abruptly.")
        previous_action = entry["action_id"]

    return {
        "source_candidates": candidates.get("source_secondary"),
        "library_version": candidates.get("library_version"),
        "arranger": {
            "name": "draft_policy_arranger",
            "description": "Rule-guided draft arrangement over candidate slots. Replaceable by LLM arranger.",
        },
        "entries": entries,
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    lines = [
        "# Draft Callbook",
        "",
        "This is a first arrangement draft. High-risk calls should be validated by live style and fandom rules.",
        "",
        "| Time | Slot | Primary | Risk | Alternatives | Notes |",
        "|---:|---|---|---|---|---|",
    ]
    for entry in result["entries"]:
        alternatives = ", ".join(a["display_name"] for a in entry["alternatives"][:3])
        notes = "<br>".join(entry["notes"])
        primary = entry["display_name"]
        if entry.get("typical_text"):
            primary += f"<br>`{entry['typical_text']}`"
        lines.append(
            f"| {entry['time']} | {entry['slot_label']} | {primary} | "
            f"{entry['risk']} | {alternatives or '-'} | {notes} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("ouen_analysis"))
    args = parser.parse_args()

    candidates = load_json(args.candidates)
    result = arrange(candidates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.candidates.name.replace(".candidates.json", "")
    json_path = args.out_dir / f"{stem}.callbook.draft.json"
    md_path = args.out_dir / f"{stem}.callbook.draft.md"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(md_path, result)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
