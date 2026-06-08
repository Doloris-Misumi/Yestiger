import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Config, Wav2Vec2Model

from callability_signal_experiment import boundary_detection_metrics_seconds, write_json
from external_structure_segmentation import (
    cosine_self_similarity,
    estimated_count,
    foote_novelty_from_similarity,
    indices_to_times,
    suppress_and_select,
)
from search_music_boundary_detectors import load_song_records, target_for


TARGETS = ("manual_fine", "manual_coarse")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def zscore_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(values, axis=0, keepdims=True)
    std = np.nanstd(values, axis=0, keepdims=True)
    return np.nan_to_num((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)


def resolve_audio_path(record: Dict[str, Any], root: Path) -> Path:
    song = record.get("annotation", {}).get("song") or {}
    raw = song.get("audio_path") or record.get("struct", {}).get("path")
    if not raw:
        raise RuntimeError(f"No audio path for {record['song_id']}")
    path = Path(str(raw))
    return path if path.is_absolute() else root / path


def find_mert_snapshot(model_dir: Path) -> Path:
    config_files = sorted(model_dir.glob("snapshots/*/config.json"))
    for config_path in config_files:
        snapshot = config_path.parent
        if (snapshot / "pytorch_model.bin").exists():
            return snapshot
    raise RuntimeError(f"No local MERT snapshot with pytorch_model.bin found under {model_dir}")


def load_mert_as_wav2vec2(snapshot: Path) -> Wav2Vec2Model:
    raw = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    raw["model_type"] = "wav2vec2"
    raw.pop("auto_map", None)
    raw.pop("_name_or_path", None)
    raw.pop("architectures", None)
    config = Wav2Vec2Config(**raw)
    model = Wav2Vec2Model(config)
    state = torch.load(snapshot / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_audio_24000(path: Path) -> torch.Tensor:
    audio, sr = torchaudio.load(str(path))
    audio = audio.mean(dim=0, keepdim=True)
    if int(sr) != 24000:
        audio = torchaudio.functional.resample(audio, int(sr), 24000)
    audio = audio.to(torch.float32)
    audio = (audio - audio.mean()) / audio.std().clamp_min(1e-6)
    return audio


def extract_bar_embeddings(
    record: Dict[str, Any],
    model: Wav2Vec2Model,
    audio_path: Path,
    chunk_seconds: float,
    device: torch.device,
) -> np.ndarray:
    rows = record["rows"]
    starts = np.asarray([float(row["start"]) for row in rows], dtype=np.float32)
    ends = np.asarray([float(row["end"]) for row in rows], dtype=np.float32)
    audio = load_audio_24000(audio_path)
    sr = 24000
    chunk_samples = max(sr, int(round(float(chunk_seconds) * sr)))
    total_samples = int(audio.shape[1])
    hidden_size = int(model.config.hidden_size)
    sums = np.zeros((len(rows), hidden_size), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.int64)
    model.to(device)

    with torch.inference_mode():
        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(total_samples, start_sample + chunk_samples)
            chunk = audio[:, start_sample:end_sample].to(device)
            if chunk.shape[1] < 400:
                continue
            output = model(chunk).last_hidden_state.detach().cpu().numpy()[0].astype(np.float32)
            if output.shape[0] == 0:
                continue
            chunk_start = start_sample / sr
            chunk_end = end_sample / sr
            frame_times = np.linspace(chunk_start, chunk_end, num=output.shape[0], endpoint=False, dtype=np.float32)
            bar_indices = np.searchsorted(starts, frame_times, side="right") - 1
            valid = (bar_indices >= 0) & (bar_indices < len(rows)) & (frame_times < ends[np.clip(bar_indices, 0, len(rows) - 1)])
            for frame_index, bar_index in enumerate(bar_indices[valid]):
                sums[int(bar_index)] += output[np.nonzero(valid)[0][frame_index]]
                counts[int(bar_index)] += 1

    embeddings = np.zeros((len(rows), hidden_size), dtype=np.float32)
    nonzero = counts > 0
    embeddings[nonzero] = (sums[nonzero] / counts[nonzero, None]).astype(np.float32)
    if not np.all(nonzero):
        available = np.where(nonzero)[0]
        for index in np.where(~nonzero)[0]:
            if available.size:
                nearest = int(available[np.argmin(np.abs(available - index))])
                embeddings[index] = embeddings[nearest]
    return zscore_columns(embeddings)


def load_or_extract_embeddings(
    record: Dict[str, Any],
    model: Wav2Vec2Model,
    root: Path,
    cache_dir: Path,
    chunk_seconds: float,
    device: torch.device,
) -> np.ndarray:
    cache_path = cache_dir / f"{record['song_id']}.mert95m_bar_embeddings.npz"
    if cache_path.exists():
        return np.load(cache_path)["embeddings"].astype(np.float32)
    audio_path = resolve_audio_path(record, root)
    print(f"Extracting MERT embeddings: {record['song_id']} <- {audio_path}", flush=True)
    embeddings = extract_bar_embeddings(record, model, audio_path, chunk_seconds, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        song_id=str(record["song_id"]),
        model="m-a-p/MERT-v1-95M loaded as standard Wav2Vec2Model without remote code",
        chunk_seconds=float(chunk_seconds),
    )
    return embeddings


def aggregate_metrics(per_song: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tp = sum(int(item["tp"]) for item in per_song)
    fp = sum(int(item["fp"]) for item in per_song)
    fn = sum(int(item["fn"]) for item in per_song)
    target_count = sum(int(item["target_count"]) for item in per_song)
    predicted_count = sum(int(item["predicted_count"]) for item in per_song)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "target_count": target_count,
        "predicted_count": predicted_count,
        "mean_tolerance_seconds": round(float(np.mean([float(item["tolerance_seconds"]) for item in per_song])), 6) if per_song else 0.0,
    }


def predict_boundaries(
    record: Dict[str, Any],
    train_records: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    candidate: Dict[str, Any],
    target_name: str,
) -> List[int]:
    sim = cosine_self_similarity(embeddings, smooth_sigma=float(candidate["smooth_sigma"]))
    novelty = foote_novelty_from_similarity(sim, kernel_size=int(candidate["kernel_size"]))
    count = estimated_count(train_records, record, target_name, float(candidate["scale"]))
    return suppress_and_select(minmax(novelty), count=count, min_gap=int(candidate["min_gap"]))


def evaluate_fixed(
    records: Sequence[Dict[str, Any]],
    embeddings_by_song: Dict[str, np.ndarray],
    candidate: Dict[str, Any],
    target_name: str,
) -> Dict[str, Any]:
    per_song = []
    predictions = []
    for record in records:
        train_records = [item for item in records if item["song_id"] != record["song_id"]]
        boundary_indices = predict_boundaries(record, train_records, embeddings_by_song[str(record["song_id"])], candidate, target_name)
        predicted = indices_to_times(record, boundary_indices)
        target = target_for(record, target_name)
        metrics = boundary_detection_metrics_seconds(target, predicted, float(record["tolerance_seconds"]))
        metrics["song_id"] = record["song_id"]
        metrics["boundary_indices"] = boundary_indices
        per_song.append(metrics)
        predictions.append(
            {
                "song_id": record["song_id"],
                "target": target_name,
                "method": candidate["method"],
                "boundary_indices": boundary_indices,
                "boundary_times": predicted,
                "target_times": target,
                "metrics": metrics,
            }
        )
    return {
        "target": target_name,
        "method": candidate["method"],
        "group": "04_mert_embedding",
        "candidate": candidate,
        "uses_allin1": False,
        "uses_manual_training": bool(candidate.get("uses_manual_training", False)),
        "aggregate": aggregate_metrics(per_song),
        "per_song": per_song,
        "predictions": predictions,
    }


def training_score(
    train_records: Sequence[Dict[str, Any]],
    embeddings_by_song: Dict[str, np.ndarray],
    candidate: Dict[str, Any],
    target_name: str,
) -> float:
    per_song = []
    for record in train_records:
        inner_train = [item for item in train_records if item["song_id"] != record["song_id"]]
        boundary_indices = predict_boundaries(record, inner_train, embeddings_by_song[str(record["song_id"])], candidate, target_name)
        predicted = indices_to_times(record, boundary_indices)
        metrics = boundary_detection_metrics_seconds(target_for(record, target_name), predicted, float(record["tolerance_seconds"]))
        per_song.append(metrics)
    return float(aggregate_metrics(per_song)["f1"])


def evaluate_loso_tuned(
    records: Sequence[Dict[str, Any]],
    embeddings_by_song: Dict[str, np.ndarray],
    grid: Sequence[Dict[str, Any]],
    target_name: str,
) -> Dict[str, Any]:
    per_song = []
    predictions = []
    selected_rows = []
    for record in records:
        train_records = [item for item in records if item["song_id"] != record["song_id"]]
        scored = [(training_score(train_records, embeddings_by_song, candidate, target_name), candidate) for candidate in grid]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_candidate = scored[0]
        boundary_indices = predict_boundaries(record, train_records, embeddings_by_song[str(record["song_id"])], best_candidate, target_name)
        predicted = indices_to_times(record, boundary_indices)
        target = target_for(record, target_name)
        metrics = boundary_detection_metrics_seconds(target, predicted, float(record["tolerance_seconds"]))
        metrics["song_id"] = record["song_id"]
        metrics["boundary_indices"] = boundary_indices
        metrics["selected_training_f1"] = round(float(best_score), 6)
        metrics["selected_method"] = best_candidate["method"]
        per_song.append(metrics)
        selected_rows.append(
            {
                "song_id": record["song_id"],
                "target": target_name,
                "selected_training_f1": round(float(best_score), 6),
                "selected_candidate": best_candidate,
            }
        )
        predictions.append(
            {
                "song_id": record["song_id"],
                "target": target_name,
                "method": "mert95m_contextual_foote_loso_tuned",
                "selected_method": best_candidate["method"],
                "boundary_indices": boundary_indices,
                "boundary_times": predicted,
                "target_times": target,
                "metrics": metrics,
            }
        )
    return {
        "target": target_name,
        "method": "mert95m_contextual_foote_loso_tuned",
        "group": "04_mert_embedding",
        "candidate": {
            "model": "m-a-p/MERT-v1-95M",
            "load_mode": "standard Wav2Vec2Model, no trust_remote_code",
            "decoder": "Foote-style self-similarity novelty on bar-pooled contextual embeddings",
        },
        "uses_allin1": False,
        "uses_manual_training": True,
        "aggregate": aggregate_metrics(per_song),
        "per_song": per_song,
        "selected_candidates": selected_rows,
        "predictions": predictions,
    }


def fixed_candidate() -> Dict[str, Any]:
    return {
        "method": "mert95m_contextual_foote_fixed",
        "model": "m-a-p/MERT-v1-95M",
        "kernel_size": 8,
        "smooth_sigma": 0.5,
        "scale": 1.0,
        "min_gap": 2,
        "uses_manual_training": True,
    }


def candidate_grid() -> List[Dict[str, Any]]:
    candidates = []
    for kernel_size in (4, 6, 8, 10, 12):
        for smooth_sigma in (0.0, 0.5, 1.0):
            for scale in (0.85, 1.0, 1.15):
                for min_gap in (1, 2, 3):
                    candidates.append(
                        {
                            "method": (
                                f"mert95m_foote_k{kernel_size}_smooth{str(smooth_sigma).replace('.', 'p')}_"
                                f"scale{str(scale).replace('.', 'p')}_gap{min_gap}"
                            ),
                            "model": "m-a-p/MERT-v1-95M",
                            "kernel_size": kernel_size,
                            "smooth_sigma": smooth_sigma,
                            "scale": scale,
                            "min_gap": min_gap,
                            "uses_manual_training": True,
                        }
                    )
    return candidates


def summary_lines(results: Sequence[Dict[str, Any]]) -> List[str]:
    lines = [
        "# MERT Embedding Structure Segmentation",
        "",
        "MERT-v1-95M was loaded from local Hugging Face weights as a standard Wav2Vec2Model with exact state-dict match. No remote repository code was executed.",
        "",
        "| Target | Method | Precision | Recall | F1 | Target Bnd | Pred Bnd |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(results, key=lambda item: (item["target"], -float(item["aggregate"]["f1"])))
    for item in ordered:
        metrics = item["aggregate"]
        lines.append(
            f"| {item['target']} | {item['method']} | {metrics['precision']:.3f} | "
            f"{metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['target_count']} | {metrics['predicted_count']} |"
        )
    return lines


def write_artifacts(out_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    write_json(out_dir / "mert_structure_metrics.json", {"results": results})
    (out_dir / "mert_structure_summary.md").write_text("\n".join(summary_lines(results)) + "\n", encoding="utf-8")
    for result in results:
        method_dir = out_dir / "methods" / str(result["method"]) / str(result["target"])
        method_dir.mkdir(parents=True, exist_ok=True)
        write_json(method_dir / "metrics.json", {key: value for key, value in result.items() if key != "predictions"})
        write_jsonl(method_dir / "predictions.jsonl", result["predictions"])
        if "selected_candidates" in result:
            write_jsonl(method_dir / "selected_candidates.jsonl", result["selected_candidates"])


def save_code_snapshot(out_dir: Path) -> None:
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for path in (
        Path("scripts/mert_embedding_segmentation.py"),
        Path("scripts/external_structure_segmentation.py"),
    ):
        if path.exists():
            shutil.copy2(path, snapshot_dir / path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MERT embedding based structure segmentation.")
    parser.add_argument("--bars-dir", type=Path, default=Path("experiments/signal_callability"))
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/mert_structure_segmentation/round1"))
    parser.add_argument("--model-dir", type=Path, default=Path(".hf/hub/models--m-a-p--MERT-v1-95M"))
    parser.add_argument("--cache-dir", type=Path, default=Path("experiments/mert_structure_segmentation/cache"))
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fixed-only", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    records = load_song_records(args.bars_dir)
    if len(records) < 2:
        raise SystemExit("Need at least two songs with signal_bars outputs.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_code_snapshot(args.out_dir)

    snapshot = find_mert_snapshot(args.model_dir)
    model = load_mert_as_wav2vec2(snapshot)
    device = torch.device("cpu")
    root = Path(".").resolve()
    embeddings_by_song = {
        str(record["song_id"]): load_or_extract_embeddings(record, model, root, args.cache_dir, args.chunk_seconds, device)
        for record in records
    }

    results: List[Dict[str, Any]] = []
    if not args.skip_fixed:
        candidate = fixed_candidate()
        for target_name in TARGETS:
            print(f"Evaluating {target_name}: {candidate['method']}", flush=True)
            results.append(evaluate_fixed(records, embeddings_by_song, candidate, target_name))
            write_artifacts(args.out_dir, results)

    if not args.fixed_only:
        grid = candidate_grid()
        for target_name in TARGETS:
            print(f"Evaluating {target_name}: mert95m_contextual_foote_loso_tuned", flush=True)
            results.append(evaluate_loso_tuned(records, embeddings_by_song, grid, target_name))
            write_artifacts(args.out_dir, results)

    write_artifacts(args.out_dir, results)
    print(f"Wrote summary: {args.out_dir / 'mert_structure_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
