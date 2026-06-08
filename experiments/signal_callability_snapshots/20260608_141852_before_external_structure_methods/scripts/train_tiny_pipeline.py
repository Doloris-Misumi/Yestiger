import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from build_pipeline_dataset import action_span_feature_row, load_library, rounded


BASE_NUMERIC_FEATURES = [
    "relative_pos",
    "duration",
    "bar_duration_ratio",
    "start_observed_downbeat",
    "start_extrapolated_downbeat",
    "allin1_struct_boundary",
    "allin1_struct_label_overlap",
]

ACTION_NUMERIC_FEATURES = [
    "duration_bars",
    "relative_pos",
    "action_min_bars",
    "action_max_bars",
    "action_intensity",
    "action_risk",
    "duration_fit",
    "duration_under",
    "duration_over",
    "context_overlap",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def one_hot(value: str, vocab: Sequence[str]) -> List[float]:
    return [1.0 if value == item else 0.0 for item in vocab]


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


class Standardizer:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, values: Sequence[Sequence[float]]) -> "Standardizer":
        array = np.asarray(values, dtype=np.float32)
        mean = array.mean(axis=0)
        std = array.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean.tolist(), std.tolist())

    def transform(self, values: Sequence[float]) -> List[float]:
        array = (np.asarray(values, dtype=np.float32) - self.mean) / self.std
        return array.tolist()

    def to_json(self) -> Dict[str, Any]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def group_rows_by_song(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    songs: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        songs[str(row["song_id"])].append(row)
    for song_rows in songs.values():
        song_rows.sort(key=lambda item: int(item["bar_index"]))
    return dict(sorted(songs.items()))


def base_numeric(row: Dict[str, Any]) -> List[float]:
    features = row.get("features") or {}
    return [float(features.get(name, 0.0)) for name in BASE_NUMERIC_FEATURES]


def collect_struct_vocab(rows: Sequence[Dict[str, Any]]) -> List[str]:
    return sorted({str((row.get("features") or {}).get("allin1_struct_label") or "unknown") for row in rows})


def build_base_features(
    row: Dict[str, Any],
    struct_vocab: Sequence[str],
    standardizer: Standardizer,
) -> List[float]:
    features = row.get("features") or {}
    return (
        standardizer.transform(base_numeric(row))
        + one_hot(str(features.get("allin1_struct_label") or "unknown"), struct_vocab)
    )


def build_segment_sequences(
    songs: Dict[str, List[Dict[str, Any]]],
    struct_vocab: Sequence[str],
    standardizer: Standardizer,
) -> List[Dict[str, Any]]:
    sequences = []
    for song_id, rows in songs.items():
        sequences.append(
            {
                "song_id": song_id,
                "x": torch.tensor(
                    [build_base_features(row, struct_vocab, standardizer) for row in rows],
                    dtype=torch.float32,
                ),
                "y_label": torch.tensor([int(row["target"]["music_label_id"]) for row in rows], dtype=torch.long),
                "y_boundary": torch.tensor([float(row["target"]["boundary"]) for row in rows], dtype=torch.float32),
                "rows": rows,
            }
        )
    return sequences


def build_call_sequences(
    songs: Dict[str, List[Dict[str, Any]]],
    struct_vocab: Sequence[str],
    label_vocab: Sequence[str],
    standardizer: Standardizer,
    predicted_labels: Dict[str, List[int]] = None,
) -> List[Dict[str, Any]]:
    sequences = []
    predicted_labels = predicted_labels or {}
    for song_id, rows in songs.items():
        x_values = []
        for index, row in enumerate(rows):
            if song_id in predicted_labels:
                label_id = predicted_labels[song_id][index]
                label_name = label_vocab[label_id]
            else:
                label_name = str(row["target"]["music_label"])
            x_values.append(build_base_features(row, struct_vocab, standardizer) + one_hot(label_name, label_vocab))
        sequences.append(
            {
                "song_id": song_id,
                "x": torch.tensor(x_values, dtype=torch.float32),
                "y_label": torch.tensor([int(row["target"]["call_role_id"]) for row in rows], dtype=torch.long),
                "y_boundary": torch.tensor([float(row["target"]["call_boundary"]) for row in rows], dtype=torch.float32),
                "rows": rows,
            }
        )
    return sequences


class TinySequenceTagger(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_labels: int):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.15)
        self.label_head = nn.Linear(hidden_dim * 2, num_labels)
        self.boundary_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(x.unsqueeze(0))
        encoded = self.dropout(encoded.squeeze(0))
        return self.label_head(encoded), self.boundary_head(encoded).squeeze(-1)


def train_sequence_model(
    sequences: Sequence[Dict[str, Any]],
    num_labels: int,
    epochs: int,
    lr: float,
    hidden_dim: int,
) -> TinySequenceTagger:
    input_dim = int(sequences[0]["x"].shape[-1])
    model = TinySequenceTagger(input_dim=input_dim, hidden_dim=hidden_dim, num_labels=num_labels)
    label_counts = Counter()
    boundary_values = []
    for seq in sequences:
        label_counts.update(seq["y_label"].tolist())
        boundary_values.extend(seq["y_boundary"].tolist())

    class_weights = torch.ones(num_labels, dtype=torch.float32)
    total = sum(label_counts.values())
    for label_id in range(num_labels):
        if label_counts[label_id]:
            class_weights[label_id] = total / (num_labels * label_counts[label_id])

    positives = sum(boundary_values)
    negatives = len(boundary_values) - positives
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        random_order = list(sequences)
        random.shuffle(random_order)
        for seq in random_order:
            model.train()
            optimizer.zero_grad()
            label_logits, boundary_logits = model(seq["x"])
            loss = ce_loss(label_logits, seq["y_label"]) + 0.45 * bce_loss(boundary_logits, seq["y_boundary"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
    return model


def eval_sequence_model(model: TinySequenceTagger, sequences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    y_boundary: List[int] = []
    p_boundary: List[int] = []
    song_metrics = {}
    with torch.no_grad():
        for seq in sequences:
            label_logits, boundary_logits = model(seq["x"])
            labels = label_logits.argmax(dim=-1)
            boundary = (torch.sigmoid(boundary_logits) >= 0.5).long()
            y = seq["y_label"]
            b = seq["y_boundary"].long()
            total += int(y.numel())
            correct += int((labels == y).sum().item())
            y_boundary.extend(b.tolist())
            p_boundary.extend(boundary.tolist())
            song_metrics[seq["song_id"]] = {
                "label_accuracy": safe_div(int((labels == y).sum().item()), int(y.numel())),
                "boundary_f1": binary_metrics(b.tolist(), boundary.tolist())["f1"],
            }
    return {
        "label_accuracy": safe_div(correct, total),
        "boundary": binary_metrics(y_boundary, p_boundary),
        "per_song": song_metrics,
    }


def predict_sequence_labels(model: TinySequenceTagger, sequences: Sequence[Dict[str, Any]]) -> Dict[str, List[int]]:
    predictions: Dict[str, List[int]] = {}
    model.eval()
    with torch.no_grad():
        for seq in sequences:
            label_logits, _ = model(seq["x"])
            predictions[seq["song_id"]] = label_logits.argmax(dim=-1).tolist()
    return predictions


class ActionRanker(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def action_numeric(row: Dict[str, Any]) -> List[float]:
    features = row.get("features") or {}
    return [float(features.get(name, 0.0)) for name in ACTION_NUMERIC_FEATURES]


def build_action_vector(
    row: Dict[str, Any],
    action_standardizer: Standardizer,
    call_role_vocab: Sequence[str],
    label_vocab: Sequence[str],
    struct_vocab: Sequence[str],
    action_vocab: Sequence[str],
) -> List[float]:
    return (
        action_standardizer.transform(action_numeric(row))
        + one_hot(str(row.get("call_role") or "unknown"), call_role_vocab)
        + one_hot(str(row.get("music_label") or "unknown"), label_vocab)
        + one_hot(str(row.get("allin1_struct_label") or "unknown"), struct_vocab)
        + one_hot(str(row.get("action_id") or "unknown"), action_vocab)
    )


def train_action_ranker(
    rows: Sequence[Dict[str, Any]],
    action_standardizer: Standardizer,
    call_role_vocab: Sequence[str],
    label_vocab: Sequence[str],
    struct_vocab: Sequence[str],
    action_vocab: Sequence[str],
    epochs: int,
    lr: float,
) -> ActionRanker:
    x = torch.tensor(
        [
            build_action_vector(row, action_standardizer, call_role_vocab, label_vocab, struct_vocab, action_vocab)
            for row in rows
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([float(row["target"]) for row in rows], dtype=torch.float32)
    model = ActionRanker(input_dim=int(x.shape[-1]))
    positives = float(y.sum().item())
    negatives = float(y.numel() - positives)
    pos_weight = torch.tensor([max(1.0, negatives / max(1.0, positives))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
    return model


def eval_action_ranker(
    model: ActionRanker,
    rows: Sequence[Dict[str, Any]],
    action_standardizer: Standardizer,
    call_role_vocab: Sequence[str],
    label_vocab: Sequence[str],
    struct_vocab: Sequence[str],
    action_vocab: Sequence[str],
) -> Dict[str, Any]:
    x = torch.tensor(
        [
            build_action_vector(row, action_standardizer, call_role_vocab, label_vocab, struct_vocab, action_vocab)
            for row in rows
        ],
        dtype=torch.float32,
    )
    y = [int(row["target"]) for row in rows]
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x)).tolist()
    pred = [1 if score >= 0.5 else 0 for score in scores]
    grouped: Dict[str, List[Tuple[float, int, str]]] = defaultdict(list)
    for row, score in zip(rows, scores):
        grouped[str(row["group_id"])].append((float(score), int(row["target"]), str(row["action_id"])))

    top1_hits = 0
    top3_hits = 0
    ranks = []
    for candidates in grouped.values():
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        top1_hits += 1 if ranked and ranked[0][1] == 1 else 0
        top3_hits += 1 if any(item[1] == 1 for item in ranked[:3]) else 0
        positive_ranks = [index + 1 for index, item in enumerate(ranked) if item[1] == 1]
        if positive_ranks:
            ranks.append(min(positive_ranks))
    return {
        "pair_accuracy": safe_div(sum(1 for a, b in zip(y, pred) if a == b), len(y)),
        "binary": binary_metrics(y, pred),
        "top1_hit_rate": safe_div(top1_hits, len(grouped)),
        "top3_hit_rate": safe_div(top3_hits, len(grouped)),
        "mean_positive_rank": float(np.mean(ranks)) if ranks else math.nan,
        "groups": len(grouped),
    }


def merge_bars(
    rows: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    boundary_probs: Sequence[float],
    boundary_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    spans = []
    start = float(rows[0]["start"])
    current = labels[0]
    for index in range(1, len(rows)):
        should_split = labels[index] != current or boundary_probs[index] >= boundary_threshold
        if should_split:
            spans.append({"start": rounded(start), "end": rounded(float(rows[index]["start"])), "label": current})
            start = float(rows[index]["start"])
            current = labels[index]
    spans.append({"start": rounded(start), "end": rounded(float(rows[-1]["end"])), "label": current})
    return [span for span in spans if span["end"] > span["start"]]


def majority(values: Sequence[str]) -> str:
    if not values:
        return "unknown"
    return Counter(values).most_common(1)[0][0]


def score_actions_for_span(
    model: ActionRanker,
    library_actions: Dict[str, Dict[str, Any]],
    span: Dict[str, Any],
    action_standardizer: Standardizer,
    call_role_vocab: Sequence[str],
    label_vocab: Sequence[str],
    struct_vocab: Sequence[str],
    action_vocab: Sequence[str],
    limit: Optional[int] = 5,
) -> List[Dict[str, Any]]:
    role = str(span.get("call_role") or "unknown")
    if role == "keepspace":
        return []
    candidates = [
        action_id
        for action_id, action in library_actions.items()
        if action.get("category") == role and action_id in action_vocab
    ]
    pair_rows = []
    for action_id in candidates:
        pair_rows.append(
            action_span_feature_row(
                song_id=str(span.get("song_id") or "prediction"),
                span_index=int(span.get("span_index") or 0),
                span={
                    "call_role": role,
                    "recommended_actions": [],
                },
                action_id=action_id,
                action=library_actions[action_id],
                target=0,
                music_label=str(span.get("music_label") or "unknown"),
                struct_label=str(span.get("allin1_struct_label") or "unknown"),
                span_start=float(span["start"]),
                span_end=float(span["end"]),
                song_end=float(span.get("song_end") or span["end"]),
                duration_bars=float(span.get("duration_bars") or 0.0),
            )
        )
    if not pair_rows:
        return []
    x = torch.tensor(
        [
            build_action_vector(row, action_standardizer, call_role_vocab, label_vocab, struct_vocab, action_vocab)
            for row in pair_rows
        ],
        dtype=torch.float32,
    )
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(x)).tolist()
    ranked = sorted(zip(pair_rows, scores), key=lambda item: item[1], reverse=True)
    ranked_rows = ranked[:limit] if limit is not None else ranked
    return [
        {"action_id": row["action_id"], "score": round(float(score), 4)}
        for row, score in ranked_rows
    ]


def build_train_predictions(
    segmenter: TinySequenceTagger,
    call_slotter: TinySequenceTagger,
    action_ranker: ActionRanker,
    segment_sequences: Sequence[Dict[str, Any]],
    call_sequences: Sequence[Dict[str, Any]],
    label_vocab: Sequence[str],
    call_role_vocab: Sequence[str],
    action_standardizer: Standardizer,
    action_struct_vocab: Sequence[str],
    action_vocab: Sequence[str],
    library_actions: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    predictions: Dict[str, Any] = {}
    segmenter.eval()
    call_slotter.eval()
    with torch.no_grad():
        for seg_seq, call_seq in zip(segment_sequences, call_sequences):
            rows = seg_seq["rows"]
            label_logits, boundary_logits = segmenter(seg_seq["x"])
            music_ids = label_logits.argmax(dim=-1).tolist()
            music_labels = [label_vocab[index] for index in music_ids]
            music_boundary_probs = torch.sigmoid(boundary_logits).tolist()

            call_logits, call_boundary_logits = call_slotter(call_seq["x"])
            call_ids = call_logits.argmax(dim=-1).tolist()
            call_roles = [call_role_vocab[index] for index in call_ids]
            call_boundary_probs = torch.sigmoid(call_boundary_logits).tolist()

            segment_spans = merge_bars(rows, music_labels, music_boundary_probs)
            call_spans = merge_bars(rows, call_roles, call_boundary_probs)
            enriched_call_spans = []
            song_end = float(rows[-1]["end"]) if rows else 0.0
            for index, span in enumerate(call_spans):
                role = span["label"]
                covered = [
                    i
                    for i, row in enumerate(rows)
                    if min(float(row["end"]), span["end"]) > max(float(row["start"]), span["start"])
                ]
                span_music = majority([music_labels[i] for i in covered])
                span_struct = majority([str((rows[i].get("features") or {}).get("allin1_struct_label") or "unknown") for i in covered])
                duration = max(0.0, float(span["end"]) - float(span["start"]))
                bar_seconds = np.median([float(row["features"]["duration"]) for row in rows if row.get("bar_kind") == "full_bar"])
                scoring_span = {
                    "song_id": seg_seq["song_id"],
                    "span_index": index,
                    "start": span["start"],
                    "end": span["end"],
                    "song_end": song_end,
                    "duration_bars": duration / bar_seconds if bar_seconds else 0.0,
                    "call_role": role,
                    "music_label": span_music,
                    "allin1_struct_label": span_struct,
                }
                ranked_actions = score_actions_for_span(
                    action_ranker,
                    library_actions,
                    scoring_span,
                    action_standardizer,
                    call_role_vocab,
                    label_vocab,
                    action_struct_vocab,
                    action_vocab,
                )
                enriched_call_spans.append(
                    {
                        "start": span["start"],
                        "end": span["end"],
                        "call_role": role,
                        "music_label": span_music,
                        "recommended_actions": [item["action_id"] for item in ranked_actions[:1]],
                        "action_candidates": ranked_actions[:3],
                    }
                )
            predictions[seg_seq["song_id"]] = {
                "segments": [
                    {"start": item["start"], "end": item["end"], "music_label": item["label"]}
                    for item in segment_spans
                ],
                "call_spans": enriched_call_spans,
            }
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tiny YesTiger segment/call/action pipeline on bar-level data.")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/pipeline"))
    parser.add_argument("--library", type=Path, default=Path("knowledge/call_mix_library.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("models/tiny_pipeline"))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--action-epochs", type=int, default=280)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    label_vocab = list(manifest["label_vocab"])
    call_role_vocab = list(manifest["call_role_vocab"])
    rows = read_jsonl(args.data_dir / "bar_rows.jsonl")
    action_rows = read_jsonl(args.data_dir / "action_pairs.jsonl")
    songs = group_rows_by_song(rows)
    struct_vocab = collect_struct_vocab(rows)
    base_standardizer = Standardizer.fit([base_numeric(row) for row in rows])

    segment_sequences = build_segment_sequences(songs, struct_vocab, base_standardizer)
    segmenter = train_sequence_model(
        segment_sequences,
        num_labels=len(label_vocab),
        epochs=args.epochs,
        lr=0.01,
        hidden_dim=args.hidden_dim,
    )
    segment_metrics = eval_sequence_model(segmenter, segment_sequences)
    predicted_music_ids = predict_sequence_labels(segmenter, segment_sequences)

    call_sequences_oracle = build_call_sequences(songs, struct_vocab, label_vocab, base_standardizer)
    call_slotter = train_sequence_model(
        call_sequences_oracle,
        num_labels=len(call_role_vocab),
        epochs=args.epochs,
        lr=0.01,
        hidden_dim=args.hidden_dim,
    )
    call_metrics_oracle = eval_sequence_model(call_slotter, call_sequences_oracle)
    call_sequences_predicted = build_call_sequences(
        songs,
        struct_vocab,
        label_vocab,
        base_standardizer,
        predicted_labels=predicted_music_ids,
    )
    call_metrics_predicted = eval_sequence_model(call_slotter, call_sequences_predicted)

    action_struct_vocab = sorted({str(row.get("allin1_struct_label") or "unknown") for row in action_rows})
    action_vocab = list(manifest["action_vocab"])
    action_standardizer = Standardizer.fit([action_numeric(row) for row in action_rows])
    action_ranker = train_action_ranker(
        action_rows,
        action_standardizer,
        call_role_vocab,
        label_vocab,
        action_struct_vocab,
        action_vocab,
        epochs=args.action_epochs,
        lr=0.01,
    )
    action_metrics = eval_action_ranker(
        action_ranker,
        action_rows,
        action_standardizer,
        call_role_vocab,
        label_vocab,
        action_struct_vocab,
        action_vocab,
    )

    library_actions = load_library(args.library)
    predictions = build_train_predictions(
        segmenter,
        call_slotter,
        action_ranker,
        segment_sequences,
        call_sequences_predicted,
        label_vocab,
        call_role_vocab,
        action_standardizer,
        action_struct_vocab,
        action_vocab,
        library_actions,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(segmenter.state_dict(), args.out_dir / "segmenter.pt")
    torch.save(call_slotter.state_dict(), args.out_dir / "call_slotter.pt")
    torch.save(action_ranker.state_dict(), args.out_dir / "action_ranker.pt")
    metadata = {
        "label_vocab": label_vocab,
        "call_role_vocab": call_role_vocab,
        "struct_vocab": struct_vocab,
        "action_struct_vocab": action_struct_vocab,
        "action_vocab": action_vocab,
        "base_numeric_features": BASE_NUMERIC_FEATURES,
        "action_numeric_features": ACTION_NUMERIC_FEATURES,
        "base_standardizer": base_standardizer.to_json(),
        "action_standardizer": action_standardizer.to_json(),
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
    }
    metrics = {
        "songs": len(songs),
        "bar_rows": len(rows),
        "action_pairs": len(action_rows),
        "segmenter_train": segment_metrics,
        "call_slotter_train_oracle_segments": call_metrics_oracle,
        "call_slotter_train_predicted_segments": call_metrics_predicted,
        "action_ranker_train": action_metrics,
    }
    write_json(args.out_dir / "metadata.json", metadata)
    write_json(args.out_dir / "metrics.json", metrics)
    write_json(args.out_dir / "train_predictions.json", predictions)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote tiny pipeline artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
