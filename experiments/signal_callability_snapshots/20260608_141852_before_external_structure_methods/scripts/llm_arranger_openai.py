import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_OPENROUTER_CODES = {408, 429, 500, 502, 503, 504, 520, 522, 524}


OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "entries": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "time": {"type": "string"},
                    "slot_label": {"type": "string"},
                    "arrangement_type": {
                        "type": "string",
                        "enum": ["single", "chain", "support", "silence"]
                    },
                    "primary_action_id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "intensity": {"type": "number"},
                    "risk": {"type": "string"},
                    "instructions": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "sub_actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "time": {"type": "string"},
                                "action_id": {"type": "string"},
                                "display_name": {"type": "string"},
                                "text": {"type": "string"},
                                "note": {"type": "string"}
                            },
                            "required": ["start", "end", "time", "action_id", "display_name", "text", "note"]
                        }
                    },
                    "alternatives": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "rationale": {"type": "string"},
                    "cautions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "start",
                    "end",
                    "time",
                    "slot_label",
                    "arrangement_type",
                    "primary_action_id",
                    "display_name",
                    "intensity",
                    "risk",
                    "instructions",
                    "sub_actions",
                    "alternatives",
                    "rationale",
                    "cautions"
                ]
            }
        }
    },
    "required": ["summary", "entries"]
}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_api_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:05.2f}"


def parse_time_text(value: str) -> Optional[float]:
    text = value.strip()
    if not text:
        return None
    if "-" in text:
        text = text.split("-", 1)[0].strip()
    try:
        return float(text)
    except ValueError:
        pass

    parts = text.split(":")
    try:
        if len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60.0 + seconds
        if len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600.0 + minutes * 60.0 + seconds
    except ValueError:
        return None
    return None


def seconds_value(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parsed = parse_time_text(value)
        if parsed is not None:
            return parsed
    return float(fallback)


def number_value(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def slim_candidates(candidates: Dict[str, Any], top_n: int) -> Dict[str, Any]:
    slots = []
    for slot in candidates["slots"]:
        slim = {
            "start": slot["start"],
            "end": slot["end"],
            "time": slot["time"],
            "duration_seconds": slot["duration_seconds"],
            "estimated_bars": slot["estimated_bars"],
            "source_label": slot["source_label"],
            "parent_label": slot["parent_label"],
            "segment_confidence": slot["segment_confidence"],
            "context_tags": slot["context_tags"],
            "source_reason": slot["source_reason"],
            "candidates": slot.get("candidates", [])[:top_n],
            "chain_candidates": slot.get("chain_candidates", [])[:top_n],
        }
        slots.append(slim)
    return {
        "source_audio": candidates.get("source_audio"),
        "library_version": candidates.get("library_version"),
        "bar_seconds_estimate": candidates.get("bar_seconds_estimate"),
        "slots": slots,
    }


def compact_candidates(candidates: Dict[str, Any], top_n: int) -> Dict[str, Any]:
    slots = []
    for index, slot in enumerate(candidates["slots"]):
        slots.append({
            "slot_index": index,
            "start": slot["start"],
            "end": slot["end"],
            "time": slot["time"],
            "bars": slot["estimated_bars"],
            "label": slot["source_label"],
            "tags": slot["context_tags"],
            "candidates": [
                {
                    "id": c["action_id"],
                    "name": c["display_name"],
                    "score": c["score"],
                    "risk": c["risk"],
                    "duration": c.get("duration", {}),
                    "text": c.get("typical_text", ""),
                }
                for c in slot.get("candidates", [])[:top_n]
            ],
            "chain_candidates": [
                {
                    "id": chain["chain_id"],
                    "name": chain["display_name"],
                    "score": chain["score"],
                    "risk": chain["risk"],
                    "actions": [
                        {
                            "id": action["action_id"],
                            "name": action["display_name"],
                            "time": action["time"],
                            "text": action.get("text", ""),
                            "risk": action["risk"],
                        }
                        for action in chain.get("actions", [])
                    ],
                }
                for chain in slot.get("chain_candidates", [])[:top_n]
            ],
        })
    return {
        "source_audio": candidates.get("source_audio"),
        "bar_seconds_estimate": candidates.get("bar_seconds_estimate"),
        "slot_count": len(slots),
        "slots": slots,
    }


def build_prompt(candidates: Dict[str, Any], profile: Dict[str, Any]) -> List[Dict[str, str]]:
    system = (
        "You are YesTiger's LLM Arranger, an expert idol/anison live callbook arranger. "
        "You do not detect beats or audio boundaries. You only arrange from provided candidate slots. "
        "Optimize for live usability: avoid overfilling vocals, use call density intentionally, "
        "split long instrumental slots into sub-actions when appropriate, and mark risky calls clearly. "
        "Return only JSON that follows the provided schema."
    )

    user = {
        "task": "Arrange a timestamped callbook from candidate slots.",
        "style_profile": profile,
        "arrangement_rules": [
            "Use only action IDs that appear in each slot's candidates, except keep_space for silence.",
            "For long instrumental or extended instrumental slots, prefer chain arrangements if profile allows chain mix.",
            "If chain_candidates are provided, treat each as a suggested packing of multiple atomic actions.",
            "To choose a chain_candidate, set arrangement_type to chain and copy its actions into sub_actions.",
            "kaho_sanren_mix / Variable Triple MIX is one atomic MIX action. Do not split it into repeated kaho_sanren_mix sub-actions.",
            "When making a chain, use distinct action IDs or clearly different actions instead of repeating the same atomic MIX.",
            "keep_space is a last-resort choice. If a slot has candidates, choose one unless there is a clear reason to leave space.",
            "Do not return an all-keep_space or mostly-keep_space callbook for this profile.",
            "For verse_subsection, usually keep support light: PPPH, clap, or silence.",
            "For pre_chorus candidates, Ietora is allowed only if the profile allows it; otherwise choose PPPH/clap.",
            "For chorus, prefer repeatable support such as Fuwa Fuwa, Hai Hai, or Clap.",
            "Do not blindly choose the highest score. Consider density, risk, and musical continuity.",
            "If a slot is too short for a full MIX, use a shortened cue or choose a rhythm call.",
            "If uncertainty is high, include cautions rather than pretending certainty.",
            "Use sub_actions for chains inside a long slot; align sub_actions inside the parent start/end.",
            "You must create one entry for every provided slot. Do not return an empty entries array."
        ],
        "candidates": candidates,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)}
    ]


def extract_output_text(response: Dict[str, Any]) -> str:
    if "output_text" in response:
        return response["output_text"]
    parts: List[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and "text" in content:
                parts.append(content["text"])
    return "\n".join(parts)


def default_api_config_path(provider: str) -> Path:
    if provider == "openrouter":
        return Path("config/openrouter.local.json")
    return Path("config/openai.local.json")


def resolve_api_settings(args: argparse.Namespace) -> Dict[str, Any]:
    provider = args.provider
    api_config = args.api_config or default_api_config_path(provider)
    config = load_api_config(api_config)

    if provider == "openrouter":
        env_key = os.environ.get("OPENROUTER_API_KEY")
        placeholder = "PASTE_YOUR_OPENROUTER_API_KEY_HERE"
        default_model = "openrouter/free"
        default_url = DEFAULT_OPENROUTER_API_URL
    else:
        env_key = os.environ.get("OPENAI_API_KEY")
        placeholder = "PASTE_YOUR_OPENAI_API_KEY_HERE"
        default_model = "gpt-5.2"
        default_url = DEFAULT_OPENAI_API_URL

    api_key = env_key or config.get("api_key")
    if api_key == placeholder:
        api_key = None

    return {
        "provider": provider,
        "api_config": str(api_config),
        "api_key": api_key,
        "model": args.model or config.get("model") or default_model,
        "base_url": args.base_url or config.get("base_url") or default_url,
        "timeout": args.timeout or int(config.get("timeout", 180)),
        "retries": args.retries if args.retries is not None else int(config.get("retries", 2)),
        "structured_output": not args.no_structured_output,
        "site_url": config.get("site_url", "http://localhost"),
        "app_name": config.get("app_name", "YesTiger"),
    }


def call_openai_responses(api_settings: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    api_key = api_settings.get("api_key")
    if not api_key:
        raise RuntimeError(
            "OpenAI API key is not configured. Put it in config/openai.local.json "
            "or set OPENAI_API_KEY in the current shell."
        )

    payload = {
        "model": api_settings["model"],
        "input": messages,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "yes_tiger_callbook",
                "strict": True,
                "schema": OUTPUT_SCHEMA
            }
        }
    }

    response = requests.post(
        api_settings["base_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=api_settings["timeout"],
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {response.status_code}: {response.text}")

    raw = response.json()
    text = extract_output_text(raw)
    if not text:
        raise RuntimeError(f"Could not find output text in response: {json.dumps(raw)[:1000]}")
    return json.loads(text)


def extract_chat_content(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def parse_json_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def call_openrouter_chat_once(api_settings: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    api_key = api_settings.get("api_key")
    if not api_key:
        raise RuntimeError(
            "OpenRouter API key is not configured. Put it in config/openrouter.local.json "
            "or set OPENROUTER_API_KEY in the current shell."
        )

    payload = {
        "model": api_settings["model"],
        "messages": messages,
        "temperature": 0.2,
    }
    if api_settings.get("structured_output", True):
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "yes_tiger_callbook",
                "strict": True,
                "schema": OUTPUT_SCHEMA
            }
        }
        payload["provider"] = {
            "require_parameters": True
        }
    else:
        payload["messages"] = [
            messages[0],
            {
                "role": "system",
                "content": (
                    "Return raw JSON only. Do not wrap it in Markdown. "
                    "The JSON object must contain summary and entries."
                ),
            },
            messages[1],
        ]

    response = requests.post(
        api_settings["base_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": api_settings.get("site_url", "http://localhost"),
            "X-Title": api_settings.get("app_name", "YesTiger"),
        },
        json=payload,
        timeout=api_settings["timeout"],
    )
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter API error {response.status_code}: {response.text}")

    raw = response.json()
    if "error" in raw:
        error = raw["error"]
        code = error.get("code", "unknown")
        message = error.get("message", json.dumps(error, ensure_ascii=False))
        raise RuntimeError(f"OpenRouter API error {code}: {message}")

    text = extract_chat_content(raw)
    if not text:
        raise RuntimeError(f"Could not find chat content in response: {json.dumps(raw)[:1000]}")
    return parse_json_text(text)


def call_openrouter_chat(api_settings: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    attempts = max(1, int(api_settings.get("retries", 2)) + 1)
    last_error: Optional[RuntimeError] = None
    for attempt in range(1, attempts + 1):
        try:
            return call_openrouter_chat_once(api_settings, messages)
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            retryable = any(f"error {code}" in message for code in RETRYABLE_OPENROUTER_CODES)
            if not retryable or attempt == attempts:
                break
            wait_seconds = min(20, 2 ** attempt)
            print(f"OpenRouter temporary error on attempt {attempt}/{attempts}: {message}")
            print(f"Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)
    assert last_error is not None
    raise RuntimeError(
        f"{last_error}\n"
        "This is usually an upstream OpenRouter/free-model timeout. Try rerunning, "
        "or add --no-structured-output, --top-n 3, or choose a specific free model."
    )


def call_llm(api_settings: Dict[str, Any], messages: List[Dict[str, str]]) -> Dict[str, Any]:
    if api_settings["provider"] == "openrouter":
        return call_openrouter_chat(api_settings, messages)
    return call_openai_responses(api_settings, messages)


def validate_arrangement(result: Dict[str, Any], expected_entries: int) -> None:
    entries = result.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            "LLM returned no callbook entries. Try a specific OpenRouter model instead of openrouter/free, "
            "or rerun with a smaller --top-n."
        )
    if len(entries) < max(1, expected_entries // 2):
        raise RuntimeError(
            f"LLM returned too few entries ({len(entries)} for {expected_entries} slots). "
            "Try a stronger model or rerun with --top-n 3."
        )


def action_id_from_candidate(candidate: Dict[str, Any]) -> str:
    return candidate.get("action_id") or candidate.get("id") or ""


def coerce_action_id(value: Any) -> str:
    if isinstance(value, dict):
        return value.get("action_id") or value.get("id") or value.get("action") or "keep_space"
    if value is None:
        return "keep_space"
    return str(value)


def action_name_from_candidate(candidate: Dict[str, Any]) -> str:
    return candidate.get("display_name") or candidate.get("name") or action_id_from_candidate(candidate)


def action_text_from_candidate(candidate: Dict[str, Any]) -> str:
    return candidate.get("typical_text") or candidate.get("text") or ""


def find_candidate(slot: Dict[str, Any], action_id: str) -> Dict[str, Any]:
    action_id = coerce_action_id(action_id)
    for candidate in slot.get("candidates", []):
        if action_id_from_candidate(candidate) == action_id:
            return candidate
    if action_id == "keep_space":
        return {"id": "keep_space", "name": "Keep Space", "risk": "low", "text": ""}
    return {"id": action_id, "name": action_id, "risk": "medium", "text": ""}


def find_chain_candidate(slot: Dict[str, Any], chain_id: str) -> Optional[Dict[str, Any]]:
    for chain in slot.get("chain_candidates", []):
        if chain.get("chain_id") == chain_id or chain.get("id") == chain_id:
            return chain
    return None


def stringify_list(values: Any) -> List[str]:
    if not values:
        return []
    if isinstance(values, str):
        text = values.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                return stringify_list([json.loads(text)])
            except json.JSONDecodeError:
                pass
        return [values]
    if not isinstance(values, list):
        values = [values]
    result = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    result.extend(stringify_list([json.loads(text)]))
                    continue
                except json.JSONDecodeError:
                    pass
            result.append(value)
        elif isinstance(value, dict) and ("text" in value or "name" in value):
            name = value.get("name") or value.get("display_name") or value.get("id") or "Action"
            text = value.get("text") or ""
            result.append(f"{name}: {text}" if text else str(name))
        else:
            result.append(json.dumps(value, ensure_ascii=False))
    return result


def normalized_sub_actions(raw_entry: Dict[str, Any], slot: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_subs = raw_entry.get("sub_actions") or []
    if not isinstance(raw_subs, list):
        return []

    slot_start = float(slot["start"])
    slot_end = float(slot["end"])
    span = max(0.01, slot_end - slot_start)
    count = max(1, len(raw_subs))
    normalized: List[Dict[str, Any]] = []
    for index, raw_sub in enumerate(raw_subs):
        action_id = coerce_action_id(raw_sub.get("action_id") or raw_sub.get("action") or raw_sub.get("id"))
        candidate = find_candidate(slot, action_id)
        start = seconds_value(raw_sub.get("start"), slot_start + span * index / count)
        end = seconds_value(raw_sub.get("end"), slot_start + span * (index + 1) / count)
        normalized.append({
            "start": start,
            "end": end,
            "time": raw_sub.get("time") or f"{fmt_time(start)} - {fmt_time(end)}",
            "action_id": action_id,
            "display_name": raw_sub.get("display_name") or action_name_from_candidate(candidate),
            "text": raw_sub.get("text") or action_text_from_candidate(candidate),
            "note": raw_sub.get("note") or "LLM chain sub-action.",
        })
    return normalized


def sub_actions_from_chain(chain: Dict[str, Any]) -> List[Dict[str, Any]]:
    normalized = []
    for raw_action in chain.get("actions", []):
        time_text = raw_action.get("time", "")
        start = seconds_value(raw_action.get("start"), parse_time_text(time_text) or 0.0)
        end_fallback = start
        if isinstance(time_text, str) and "-" in time_text:
            parsed_end = parse_time_text(time_text.split("-", 1)[1].strip())
            if parsed_end is not None:
                end_fallback = parsed_end
        end = seconds_value(raw_action.get("end"), end_fallback)
        normalized.append({
            "start": start,
            "end": end,
            "time": raw_action.get("time") or f"{fmt_time(start)} - {fmt_time(end)}",
            "action_id": raw_action.get("action_id") or raw_action.get("id") or "keep_space",
            "display_name": raw_action.get("display_name") or raw_action.get("name") or "Action",
            "text": raw_action.get("text", ""),
            "note": "Expanded from dynamic chain candidate.",
        })
    return normalized


def fallback_entry_for_slot(slot: Dict[str, Any], reason: str) -> Dict[str, Any]:
    chain_candidates = slot.get("chain_candidates") or []
    if chain_candidates:
        chain = chain_candidates[0]
        sub_actions = sub_actions_from_chain(chain)
        chain_id = chain.get("chain_id") or chain.get("id") or "chain_candidate"
        chain_name = chain.get("display_name") or chain.get("name") or chain_id
        return {
            "start": float(slot["start"]),
            "end": float(slot["end"]),
            "time": slot.get("time") or f"{fmt_time(float(slot['start']))} - {fmt_time(float(slot['end']))}",
            "slot_label": slot.get("source_label") or slot.get("label") or "unknown",
            "arrangement_type": "chain",
            "primary_action_id": chain_id,
            "display_name": chain_name,
            "intensity": 0.75,
            "risk": chain.get("risk", "medium"),
            "instructions": [chain_name],
            "sub_actions": sub_actions,
            "alternatives": [
                candidate.get("chain_id") or candidate.get("id") or candidate.get("display_name") or candidate.get("name")
                for candidate in chain_candidates[1:4]
            ],
            "rationale": reason,
            "cautions": ["Auto-repaired from degenerate LLM output; verify by listening."],
        }

    slot_candidates = slot.get("candidates") or []
    fallback = slot_candidates[0] if slot_candidates else {
        "id": "keep_space",
        "action_id": "keep_space",
        "name": "Keep Space",
        "display_name": "Keep Space",
        "risk": "low",
        "intensity": 0.0,
        "text": "",
        "typical_text": "",
    }
    action_id = action_id_from_candidate(fallback) or "keep_space"
    display_name = action_name_from_candidate(fallback) if fallback else "Keep Space"
    text = action_text_from_candidate(fallback)
    return {
        "start": float(slot["start"]),
        "end": float(slot["end"]),
        "time": slot.get("time") or f"{fmt_time(float(slot['start']))} - {fmt_time(float(slot['end']))}",
        "slot_label": slot.get("source_label") or slot.get("label") or "unknown",
        "arrangement_type": "silence" if action_id == "keep_space" else "single",
        "primary_action_id": action_id,
        "display_name": display_name,
        "intensity": number_value(fallback.get("intensity"), 0.5) if fallback else 0.0,
        "risk": fallback.get("risk", "low") if fallback else "low",
        "instructions": [f"{display_name}: {text}" if text else display_name],
        "sub_actions": [],
        "alternatives": [],
        "rationale": reason,
        "cautions": ["Auto-filled from candidates; verify by listening."],
    }


def repair_degenerate_keep_space(entries: List[Dict[str, Any]], slots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    actionable = [
        index
        for index, slot in enumerate(slots)
        if slot.get("candidates") or slot.get("chain_candidates")
    ]
    if not actionable:
        return entries

    keep_count = sum(
        1
        for index in actionable
        if index < len(entries) and entries[index].get("primary_action_id") == "keep_space"
    )
    keep_ratio = keep_count / max(1, len(actionable))
    if keep_ratio < 0.6:
        return entries

    repaired = list(entries)
    for index in actionable:
        if index >= len(repaired):
            repaired.append(fallback_entry_for_slot(slots[index], "Filled because LLM omitted this slot."))
            continue
        if repaired[index].get("primary_action_id") == "keep_space":
            repaired[index] = fallback_entry_for_slot(
                slots[index],
                "Repaired because the LLM returned a mostly keep_space callbook.",
            )
    return repaired


def synthesize_chain_sub_actions(slot: Dict[str, Any], action_id: str, candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    action_id = coerce_action_id(action_id)
    if action_id == "kaho_sanren_mix":
        return []

    label = slot.get("source_label") or slot.get("label") or ""
    duration = float(slot["end"]) - float(slot["start"])
    should_chain = duration >= 20.0 and "inst" in label and action_id.endswith("_mix")
    if not should_chain:
        return []

    count = 3 if duration >= 26.0 else 2
    start = float(slot["start"])
    end = float(slot["end"])
    span = end - start
    display_name = action_name_from_candidate(candidate)
    text = action_text_from_candidate(candidate)
    return [
        {
            "start": start + span * index / count,
            "end": start + span * (index + 1) / count,
            "time": f"{fmt_time(start + span * index / count)} - {fmt_time(start + span * (index + 1) / count)}",
            "action_id": action_id,
            "display_name": display_name,
            "text": text,
            "note": f"Auto-split long instrumental chain phrase {index + 1}/{count}.",
        }
        for index in range(count)
    ]


def normalize_arrangement(result: Dict[str, Any], candidates: Dict[str, Any]) -> Dict[str, Any]:
    raw_entries = result.get("entries") or []
    slots = candidates.get("slots") or []
    by_slot: Dict[int, Dict[str, Any]] = {}

    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        slot_index = raw_entry.get("slot_index")
        if slot_index is None:
            for index, slot in enumerate(slots):
                if raw_entry.get("start") == slot.get("start") and raw_entry.get("end") == slot.get("end"):
                    slot_index = index
                    break
        if slot_index is None:
            continue
        slot_index = int(slot_index)
        if slot_index < 0 or slot_index >= len(slots):
            continue

        slot = slots[slot_index]
        action_id = (
            raw_entry.get("primary_action_id")
            or raw_entry.get("action_id")
            or raw_entry.get("action")
            or "keep_space"
        )
        action_id = coerce_action_id(action_id)
        chain_candidate = find_chain_candidate(slot, action_id)
        candidate = (
            {
                "id": chain_candidate["chain_id"],
                "name": chain_candidate["display_name"],
                "risk": chain_candidate.get("risk", "medium"),
                "text": chain_candidate["display_name"],
            }
            if chain_candidate
            else find_candidate(slot, action_id)
        )
        cautions = stringify_list(raw_entry.get("cautions"))
        if raw_entry.get("caution") is True:
            cautions.append("LLM marked this slot as requiring live-culture confirmation.")

        sub_actions = normalized_sub_actions(raw_entry, slot)
        if action_id == "kaho_sanren_mix" and all(
            coerce_action_id(sub.get("action_id")) == "kaho_sanren_mix"
            for sub in sub_actions
        ):
            sub_actions = []
        if chain_candidate and not sub_actions:
            sub_actions = sub_actions_from_chain(chain_candidate)
        if not sub_actions:
            sub_actions = synthesize_chain_sub_actions(slot, action_id, candidate)
        entry = by_slot.get(slot_index)
        if entry:
            entry["sub_actions"].extend(sub_actions)
            entry["cautions"].extend(caution for caution in cautions if caution not in entry["cautions"])
            continue

        raw_display_name = raw_entry.get("display_name")
        display_name = raw_display_name if isinstance(raw_display_name, str) else action_name_from_candidate(candidate)
        text = action_text_from_candidate(candidate)
        instructions = stringify_list(raw_entry.get("instructions"))
        if not instructions:
            instructions = [f"{display_name}: {text}" if text else display_name]

        arrangement_type = raw_entry.get("arrangement_type")
        if not arrangement_type:
            arrangement_type = "chain" if sub_actions or chain_candidate else ("silence" if action_id == "keep_space" else "single")

        by_slot[slot_index] = {
            "start": seconds_value(raw_entry.get("start"), float(slot["start"])),
            "end": seconds_value(raw_entry.get("end"), float(slot["end"])),
            "time": slot.get("time") or raw_entry.get("time") or f"{fmt_time(float(slot['start']))} - {fmt_time(float(slot['end']))}",
            "slot_label": slot.get("source_label") or slot.get("label") or raw_entry.get("slot_label") or "unknown",
            "arrangement_type": arrangement_type,
            "primary_action_id": action_id,
            "display_name": display_name,
            "intensity": number_value(raw_entry.get("intensity"), candidate.get("intensity", 0.5)),
            "risk": raw_entry.get("risk") or candidate.get("risk", "medium"),
            "instructions": instructions,
            "sub_actions": sub_actions,
            "alternatives": stringify_list(raw_entry.get("alternatives")) or [
                action_id_from_candidate(candidate)
                for candidate in slot.get("candidates", [])[:3]
                if action_id_from_candidate(candidate) != action_id
            ],
            "rationale": raw_entry.get("rationale") or "Normalized from LLM output and the slot candidate list.",
            "cautions": cautions,
        }

    normalized_entries = []
    for index, slot in enumerate(slots):
        if index in by_slot:
            normalized_entries.append(by_slot[index])
            continue
        normalized_entries.append(fallback_entry_for_slot(
            slot,
            "Fallback-filled because the LLM omitted this slot.",
        ))

    normalized_entries = repair_degenerate_keep_space(normalized_entries, slots)

    return {
        "summary": result.get("summary") or "LLM arrangement normalized for all candidate slots.",
        "entries": normalized_entries,
    }


def write_markdown(path: Path, result: Dict[str, Any]) -> None:
    def md_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    lines = [
        "# LLM Arranged Callbook",
        "",
        result.get("summary", ""),
        "",
        "| Time | Slot | Arrangement | Instructions | Cautions |",
        "|---:|---|---|---|---|",
    ]
    for entry in result["entries"]:
        if entry["sub_actions"]:
            arrangement = "<br>".join(
                f"{sub['time']} {sub['display_name']} `{md_text(sub['text'])}`"
                for sub in entry["sub_actions"]
            )
        else:
            arrangement = f"{entry['display_name']}"

        instructions = "<br>".join(md_text(item) for item in entry["instructions"])
        cautions = "<br>".join(md_text(item) for item in entry["cautions"]) if entry["cautions"] else "-"
        lines.append(
            f"| {entry['time']} | {entry['slot_label']} | {arrangement} | {instructions} | {cautions} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--profile", default=Path("profiles/live_style.default.json"), type=Path)
    parser.add_argument("--out-dir", default=Path("ouen_analysis"), type=Path)
    parser.add_argument("--provider", choices=["openai", "openrouter"], default="openai")
    parser.add_argument("--api-config", default=None, type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--top-n", default=6, type=int)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--timeout", default=None, type=int)
    parser.add_argument("--retries", default=None, type=int)
    parser.add_argument("--no-structured-output", action="store_true")
    parser.add_argument("--normalize-existing", default=None, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_settings = resolve_api_settings(args)
    raw_candidates = load_json(args.candidates)
    candidates = compact_candidates(raw_candidates, args.top_n) if args.compact else slim_candidates(raw_candidates, args.top_n)
    profile = load_json(args.profile)
    messages = build_prompt(candidates, profile)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.candidates.name.replace(".candidates.json", "")
    prompt_path = args.out_dir / f"{stem}.llm_prompt.json"
    json_path = args.out_dir / f"{stem}.callbook.llm.json"
    md_path = args.out_dir / f"{stem}.callbook.llm.md"

    prompt_path.write_text(json.dumps({
        "provider": api_settings["provider"],
        "model": api_settings["model"],
        "base_url": api_settings["base_url"],
        "api_config": api_settings["api_config"],
        "structured_output": api_settings["structured_output"],
        "messages": messages,
        "schema": OUTPUT_SCHEMA,
    }, indent=2, ensure_ascii=True), encoding="utf-8")

    if args.dry_run:
        print(f"Wrote {prompt_path}")
        print("Dry run only; no API request sent.")
        return

    raw_result = load_json(args.normalize_existing) if args.normalize_existing else call_llm(api_settings, messages)
    result = normalize_arrangement(raw_result, candidates)
    validate_arrangement(result, len(candidates["slots"]))
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    write_markdown(md_path, result)
    print(f"Wrote {prompt_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
