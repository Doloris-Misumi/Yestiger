import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from callability_signal_experiment import (  # noqa: E402
    ROLE_VOCAB,
    aggregate_merged_span_comparison,
    merge_role_sequence,
    merged_span_metrics,
    normalize_role,
    write_json,
    write_jsonl,
)


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_signal_rows(bars_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(bars_dir.glob("*/*.signal_bars.jsonl")):
        song_id = path.parent.name
        for row in load_jsonl(path):
            item = dict(row)
            item["_song_id"] = str(item.get("song_id") or song_id)
            rows.append(item)
    rows.sort(key=lambda item: (str(item["_song_id"]), int(item["bar_index"])))
    return rows


def group_by_song(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["_song_id"]), []).append(row)
    for song_rows in grouped.values():
        song_rows.sort(key=lambda item: int(item["bar_index"]))
    return grouped


def load_prediction_maps(path: Path) -> Dict[str, Dict[Tuple[str, int], str]]:
    maps: Dict[str, Dict[Tuple[str, int], str]] = {}
    for item in load_jsonl(path):
        method = str(item.get("method") or item.get("feature_set") or "unknown")
        key = (str(item["song_id"]), int(item["bar_index"]))
        maps.setdefault(method, {})[key] = normalize_role(item.get("prediction"))
    return maps


def roles_from_map(song_id: str, rows: Sequence[Dict[str, Any]], mapping: Dict[Tuple[str, int], str]) -> List[str]:
    return [
        normalize_role(mapping.get((song_id, int(row["bar_index"])), "keepspace"))
        for row in rows
    ]


def target_boundary_rate(train_song_rows: Sequence[Sequence[Dict[str, Any]]]) -> float:
    total_candidates = 0
    total_boundaries = 0
    for rows in train_song_rows:
        roles = [normalize_role(row["target"]["call_role"]) for row in rows]
        if len(roles) <= 1:
            continue
        total_candidates += len(roles) - 1
        total_boundaries += sum(1 for prev, current in zip(roles, roles[1:]) if prev != current)
    return total_boundaries / total_candidates if total_candidates else 0.0


def spans_from_roles(roles: Sequence[str]) -> List[Dict[str, Any]]:
    spans = []
    if not roles:
        return spans
    start = 0
    current = normalize_role(roles[0])
    for index, role in enumerate(roles[1:], start=1):
        role = normalize_role(role)
        if role == current:
            continue
        spans.append({"start": start, "end": index - 1, "role": current, "bars": index - start})
        start = index
        current = role
    spans.append({"start": start, "end": len(roles) - 1, "role": current, "bars": len(roles) - start})
    return spans


def roles_from_spans(spans: Sequence[Dict[str, Any]], n: int) -> List[str]:
    roles = ["keepspace" for _ in range(n)]
    for span in spans:
        for index in range(int(span["start"]), int(span["end"]) + 1):
            if 0 <= index < n:
                roles[index] = normalize_role(span["role"])
    return roles


def smooth_short_runs(roles: Sequence[str], min_bars: int) -> List[str]:
    spans = spans_from_roles(roles)
    if len(spans) <= 1 or min_bars <= 1:
        return [normalize_role(role) for role in roles]

    changed = True
    while changed and len(spans) > 1:
        changed = False
        candidates = [
            (int(span["bars"]), index)
            for index, span in enumerate(spans)
            if int(span["bars"]) < min_bars
        ]
        if not candidates:
            break
        _, index = min(candidates)
        left = spans[index - 1] if index > 0 else None
        current = spans[index]
        right = spans[index + 1] if index + 1 < len(spans) else None

        if left and right and left["role"] == right["role"]:
            left["end"] = right["end"]
            left["bars"] = int(left["end"]) - int(left["start"]) + 1
            spans.pop(index + 1)
            spans.pop(index)
        elif left and (not right or int(left["bars"]) >= int(right["bars"])):
            left["end"] = current["end"]
            left["bars"] = int(left["end"]) - int(left["start"]) + 1
            spans.pop(index)
        elif right:
            right["start"] = current["start"]
            right["bars"] = int(right["end"]) - int(right["start"]) + 1
            spans.pop(index)
        changed = True
    return roles_from_spans(spans, len(roles))


def majority_role(roles: Sequence[str]) -> str:
    if not roles:
        return "keepspace"
    counts = Counter(normalize_role(role) for role in roles)
    return counts.most_common(1)[0][0]


def signal_boundary_score(rows: Sequence[Dict[str, Any]], index: int, prev_len: int, next_len: int, mode: str) -> float:
    row = rows[index]
    features = row.get("features") or {}
    novelty = row.get("novelty") or {}
    signal = row.get("signal_features") or {}
    struct = float(features.get("allin1_struct_boundary", 0.0))
    fused = float(novelty.get("fused", 0.0))
    onset = float(novelty.get("onset", 0.0))
    energy_novelty = float(novelty.get("energy", 0.0))
    vocal = float(signal.get("vocal_density_proxy", 0.0))
    length_strength = min(max(min(prev_len, next_len), 0) / 4.0, 1.0)

    if mode == "len":
        return length_strength
    if mode == "novelty":
        return 0.45 * fused + 0.25 * onset + 0.20 * energy_novelty + 0.10 * struct
    if mode == "struct_novelty":
        return 0.35 * struct + 0.30 * fused + 0.20 * onset + 0.15 * energy_novelty
    if mode == "struct_novelty_len":
        signal_score = 0.35 * struct + 0.30 * fused + 0.20 * onset + 0.15 * energy_novelty
        return 0.65 * signal_score + 0.35 * length_strength
    if mode == "struct_novelty_len50":
        signal_score = 0.35 * struct + 0.30 * fused + 0.20 * onset + 0.15 * energy_novelty
        return 0.50 * signal_score + 0.50 * length_strength
    if mode == "struct_novelty_len80":
        signal_score = 0.35 * struct + 0.30 * fused + 0.20 * onset + 0.15 * energy_novelty
        return 0.80 * signal_score + 0.20 * length_strength
    if mode == "struct_heavy_len":
        signal_score = 0.50 * struct + 0.20 * fused + 0.15 * onset + 0.15 * energy_novelty
        return 0.65 * signal_score + 0.35 * length_strength
    if mode == "novelty_len":
        signal_score = 0.10 * struct + 0.45 * fused + 0.25 * onset + 0.20 * energy_novelty
        return 0.65 * signal_score + 0.35 * length_strength
    if mode == "call_window":
        signal_score = 0.28 * struct + 0.26 * fused + 0.18 * onset + 0.14 * energy_novelty + 0.14 * (1.0 - vocal)
        return 0.70 * signal_score + 0.30 * length_strength
    raise ValueError(f"Unknown boundary score mode: {mode}")


def boundary_candidates_from_spans(
    rows: Sequence[Dict[str, Any]],
    roles: Sequence[str],
    mode: str,
) -> List[Tuple[int, float]]:
    spans = spans_from_roles(roles)
    candidates = []
    for span_index in range(1, len(spans)):
        index = int(spans[span_index]["start"])
        prev_len = int(spans[span_index - 1]["bars"])
        next_len = int(spans[span_index]["bars"])
        candidates.append((index, signal_boundary_score(rows, index, prev_len, next_len, mode)))
    return candidates


def target_boundary_indices(rows: Sequence[Dict[str, Any]]) -> List[int]:
    roles = [normalize_role(row["target"]["call_role"]) for row in rows]
    return [index for index in range(1, len(roles)) if roles[index] != roles[index - 1]]


def boundary_candidate_feature_rows(
    rows: Sequence[Dict[str, Any]],
    roles: Sequence[str],
) -> List[Tuple[int, List[float]]]:
    spans = spans_from_roles(roles)
    output = []
    role_to_index = {role: index for index, role in enumerate(ROLE_VOCAB)}
    n = max(1, len(rows) - 1)
    for span_index in range(1, len(spans)):
        index = int(spans[span_index]["start"])
        prev_len = int(spans[span_index - 1]["bars"])
        next_len = int(spans[span_index]["bars"])
        row = rows[index]
        features = row.get("features") or {}
        novelty = row.get("novelty") or {}
        signal = row.get("signal_features") or {}
        prev_role = normalize_role(spans[span_index - 1]["role"])
        next_role = normalize_role(spans[span_index]["role"])
        vector = [
            index / n,
            min(prev_len / 8.0, 1.0),
            min(next_len / 8.0, 1.0),
            min(min(prev_len, next_len) / 4.0, 1.0),
            float(features.get("allin1_struct_boundary", 0.0)),
            float(features.get("allin1_struct_label_overlap", 0.0)),
            float(novelty.get("fused", 0.0)),
            float(novelty.get("timbre", 0.0)),
            float(novelty.get("harmony", 0.0)),
            float(novelty.get("energy", 0.0)),
            float(novelty.get("onset", 0.0)),
            float(signal.get("energy", 0.0)),
            float(signal.get("onset", 0.0)),
            float(signal.get("vocal_density_proxy", 0.0)),
            float(signal.get("beat_stability", 0.0)),
            signal_boundary_score(rows, index, prev_len, next_len, "struct_novelty_len"),
            signal_boundary_score(rows, index, prev_len, next_len, "call_window"),
            1.0 if prev_role == next_role else 0.0,
        ]
        vector.extend(1.0 if role_to_index[prev_role] == role_index else 0.0 for role_index in range(len(ROLE_VOCAB)))
        vector.extend(1.0 if role_to_index[next_role] == role_index else 0.0 for role_index in range(len(ROLE_VOCAB)))
        output.append((index, vector))
    return output


def boundary_model(kind: str) -> Any:
    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit("scikit-learn is required for learned boundary decoders.") from exc

    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.8, class_weight="balanced", max_iter=3000, random_state=13),
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=13,
        )
    if kind == "gb":
        return GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=4,
            subsample=0.85,
            random_state=13,
        )
    raise ValueError(f"Unknown boundary model kind: {kind}")


def roles_with_selected_boundaries(
    base_roles: Sequence[str],
    selected_boundaries: Sequence[int],
) -> List[str]:
    n = len(base_roles)
    boundaries = [0] + sorted({int(index) for index in selected_boundaries if 0 < int(index) < n}) + [n]
    decoded: List[str] = []
    previous_role = None
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        role = majority_role(base_roles[left:right])
        if previous_role is not None and role == previous_role:
            alternatives = Counter(normalize_role(item) for item in base_roles[left:right])
            alternatives.pop(role, None)
            if alternatives:
                role = alternatives.most_common(1)[0][0]
        decoded.extend([role] * (right - left))
        previous_role = role
    return decoded


def keep_topk_boundaries(
    song_id: str,
    grouped: Dict[str, List[Dict[str, Any]]],
    rows: Sequence[Dict[str, Any]],
    base_roles: Sequence[str],
    mode: str,
    rate_scale: float,
) -> List[str]:
    train_rows = [items for other_song, items in grouped.items() if other_song != song_id]
    rate = target_boundary_rate(train_rows)
    target_k = int(round(rate * max(0, len(rows) - 1) * rate_scale))
    candidates = boundary_candidates_from_spans(rows, base_roles, mode)
    if target_k <= 0 or not candidates:
        return [normalize_role(role) for role in base_roles]
    selected = sorted(candidates, key=lambda item: item[1], reverse=True)[: min(target_k, len(candidates))]
    return roles_with_selected_boundaries(base_roles, [index for index, _ in selected])


def keep_learned_topk_boundaries(
    song_id: str,
    grouped: Dict[str, List[Dict[str, Any]]],
    prediction_maps: Dict[str, Dict[Tuple[str, int], str]],
    rows: Sequence[Dict[str, Any]],
    base_roles: Sequence[str],
    base_method: str,
    model_kind: str,
    rate_scale: float,
) -> List[str]:
    x_train = []
    y_train = []
    for other_song, other_rows in grouped.items():
        if other_song == song_id:
            continue
        other_roles = roles_from_map(other_song, other_rows, prediction_maps[base_method])
        positives = target_boundary_indices(other_rows)
        for index, vector in boundary_candidate_feature_rows(other_rows, other_roles):
            x_train.append(vector)
            y_train.append(1 if any(abs(index - target) <= 1 for target in positives) else 0)

    test_candidates = boundary_candidate_feature_rows(rows, base_roles)
    if not test_candidates or len(set(y_train)) < 2:
        return keep_topk_boundaries(song_id, grouped, rows, base_roles, "struct_novelty_len", rate_scale)

    model = boundary_model(model_kind)
    model.fit(np.asarray(x_train, dtype=np.float32), np.asarray(y_train, dtype=np.int32))
    x_test = np.asarray([vector for _, vector in test_candidates], dtype=np.float32)
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_test), dtype=np.float32)[:, 1]
    else:
        scores = np.asarray(model.decision_function(x_test), dtype=np.float32)

    train_rows = [items for other_song, items in grouped.items() if other_song != song_id]
    rate = target_boundary_rate(train_rows)
    target_k = int(round(rate * max(0, len(rows) - 1) * rate_scale))
    selected = sorted(zip([index for index, _ in test_candidates], scores), key=lambda item: item[1], reverse=True)[
        : min(target_k, len(test_candidates))
    ]
    return roles_with_selected_boundaries(base_roles, [index for index, _ in selected])


def keep_threshold_boundaries(
    rows: Sequence[Dict[str, Any]],
    base_roles: Sequence[str],
    mode: str,
    threshold: float,
) -> List[str]:
    candidates = boundary_candidates_from_spans(rows, base_roles, mode)
    selected = [index for index, score in candidates if score >= threshold]
    if not selected:
        return [normalize_role(role) for role in base_roles]
    return roles_with_selected_boundaries(base_roles, selected)


def evaluate_candidate(
    grouped: Dict[str, List[Dict[str, Any]]],
    base_maps: Dict[str, Dict[Tuple[str, int], str]],
    candidate: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    method = str(candidate["method"])
    base_method = str(candidate["base_method"])
    mapping = base_maps[base_method]
    fake_results = []
    predictions = []

    for song_id, rows in grouped.items():
        base_roles = roles_from_map(song_id, rows, mapping)
        decoder = str(candidate["decoder"])
        if decoder == "raw":
            decoded_roles = base_roles
        elif decoder == "minrun":
            decoded_roles = smooth_short_runs(base_roles, int(candidate["min_bars"]))
        elif decoder == "topk":
            decoded_roles = keep_topk_boundaries(
                song_id,
                grouped,
                rows,
                base_roles,
                mode=str(candidate["score_mode"]),
                rate_scale=float(candidate["rate_scale"]),
            )
        elif decoder == "learned_topk":
            decoded_roles = keep_learned_topk_boundaries(
                song_id,
                grouped,
                base_maps,
                rows,
                base_roles,
                base_method=base_method,
                model_kind=str(candidate["boundary_model"]),
                rate_scale=float(candidate["rate_scale"]),
            )
        elif decoder == "threshold":
            decoded_roles = keep_threshold_boundaries(
                rows,
                base_roles,
                mode=str(candidate["score_mode"]),
                threshold=float(candidate["threshold"]),
            )
        else:
            raise ValueError(f"Unknown decoder: {decoder}")

        target_roles = [normalize_role(row["target"]["call_role"]) for row in rows]
        target_merged = merge_role_sequence(rows, target_roles, method="target_manual_grid")
        predicted_merged = merge_role_sequence(rows, decoded_roles, method=method)
        full_bar_durations = [
            float(row["features"]["duration"])
            for row in rows
            if row.get("bar_kind") == "full_bar"
        ]
        tolerance_seconds = float(np.median(full_bar_durations)) if full_bar_durations else 2.5
        comparison = merged_span_metrics(target_merged, predicted_merged, tolerance_seconds)
        fake_results.append({"metrics": {"merged_span_comparison": {method: comparison}}})
        for row, target, pred in zip(rows, target_roles, decoded_roles):
            predictions.append(
                {
                    "song_id": song_id,
                    "bar_index": int(row["bar_index"]),
                    "start": row["start"],
                    "end": row["end"],
                    "target": target,
                    "prediction": pred,
                    "method": method,
                    "base_method": base_method,
                    "decoder": decoder,
                }
            )

    aggregate = aggregate_merged_span_comparison(fake_results).get(method, {})
    return aggregate, predictions


def candidate_grid(base_methods: Sequence[str]) -> List[Dict[str, Any]]:
    candidates = []
    for base_method in base_methods:
        candidates.append({"method": f"{base_method}__raw", "base_method": base_method, "decoder": "raw"})
        for mode in (
            "struct_novelty_len",
            "struct_novelty_len50",
            "struct_novelty_len80",
            "struct_heavy_len",
            "novelty_len",
            "call_window",
        ):
            for scale in (1.02, 1.06, 1.1, 1.14, 1.18):
                candidates.append(
                    {
                        "method": f"{base_method}__topk_{mode}_scale{str(scale).replace('.', 'p')}",
                        "base_method": base_method,
                        "decoder": "topk",
                        "score_mode": mode,
                        "rate_scale": scale,
                    }
                )
        for boundary_model_name in ("logreg", "rf", "gb"):
            for scale in (1.0, 1.06, 1.1, 1.14):
                candidates.append(
                    {
                        "method": f"{base_method}__learned_{boundary_model_name}_scale{str(scale).replace('.', 'p')}",
                        "base_method": base_method,
                        "decoder": "learned_topk",
                        "boundary_model": boundary_model_name,
                        "rate_scale": scale,
                    }
                )
    return candidates


def summary_lines(results: Sequence[Dict[str, Any]]) -> List[str]:
    ordered = sorted(
        results,
        key=lambda item: (
            float(item["merged"].get("boundary", {}).get("f1", 0.0)),
            float(item["merged"].get("macro_role_iou", 0.0)),
            float(item["merged"].get("time_weighted_role_accuracy", 0.0)),
        ),
        reverse=True,
    )
    lines = [
        "# Boundary Decoder Search",
        "",
        "| Method | Base | Decoder | Time-W Acc | Macro IoU | Boundary P | Boundary R | Boundary F1 | Pred Bnd |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ordered:
        merged = item["merged"]
        boundary = merged.get("boundary", {})
        candidate = item["candidate"]
        lines.append(
            f"| {item['method']} | {candidate['base_method']} | {candidate['decoder']} | "
            f"{merged.get('time_weighted_role_accuracy', 0.0):.3f} | "
            f"{merged.get('macro_role_iou', 0.0):.3f} | "
            f"{boundary.get('precision', 0.0):.3f} | "
            f"{boundary.get('recall', 0.0):.3f} | "
            f"{boundary.get('f1', 0.0):.3f} | "
            f"{boundary.get('predicted_count', 0)} |"
        )
    if ordered:
        best = ordered[0]
        boundary = best["merged"].get("boundary", {})
        lines.extend(
            [
                "",
                f"Best by boundary F1: `{best['method']}` "
                f"(P={boundary.get('precision', 0.0):.3f}, "
                f"R={boundary.get('recall', 0.0):.3f}, "
                f"F1={boundary.get('f1', 0.0):.3f}).",
            ]
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Search boundary decoders from saved LOSO call-role predictions.")
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--predictions", type=Path, default=Path("experiments/signal_callability/loso_predictions.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/signal_callability_boundary_search/round1"))
    parser.add_argument(
        "--base-method",
        action="append",
        dest="base_methods",
        default=[],
        help="Base prediction method to decode. Can be passed multiple times.",
    )
    args = parser.parse_args()

    rows = load_signal_rows(args.bars_dir)
    grouped = group_by_song(rows)
    prediction_maps = load_prediction_maps(args.predictions)
    base_methods = args.base_methods or [
        "loso_audio_rf",
        "loso_audio_vote_rf1_logreg1_gb1",
    ]
    missing = [method for method in base_methods if method not in prediction_maps]
    if missing:
        raise SystemExit(f"Missing base methods in predictions file: {', '.join(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_predictions = []
    for candidate in candidate_grid(base_methods):
        print(f"Searching {candidate['method']}", flush=True)
        merged, predictions = evaluate_candidate(grouped, prediction_maps, candidate)
        result = {
            "method": candidate["method"],
            "candidate": candidate,
            "merged": merged,
        }
        results.append(result)
        all_predictions.extend(predictions)
        ordered = sorted(results, key=lambda item: float(item["merged"].get("boundary", {}).get("f1", 0.0)), reverse=True)
        write_json(args.out_dir / "boundary_search_metrics.json", {"results": ordered})
        write_jsonl(args.out_dir / "boundary_search_predictions.jsonl", all_predictions)
        (args.out_dir / "boundary_search_summary.md").write_text(
            "\n".join(summary_lines(ordered)) + "\n",
            encoding="utf-8",
        )

    results.sort(key=lambda item: float(item["merged"].get("boundary", {}).get("f1", 0.0)), reverse=True)
    write_json(args.out_dir / "boundary_search_metrics.json", {"results": results})
    write_jsonl(args.out_dir / "boundary_search_predictions.jsonl", all_predictions)
    (args.out_dir / "boundary_search_summary.md").write_text("\n".join(summary_lines(results)) + "\n", encoding="utf-8")
    print(f"Wrote boundary search summary: {args.out_dir / 'boundary_search_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
