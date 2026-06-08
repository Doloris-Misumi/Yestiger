import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def run_command(command: List[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_audio_from_struct(struct_path: Path) -> Optional[Path]:
    if not struct_path.exists():
        return None
    data = load_json(struct_path)
    raw_path = data.get("path")
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else (struct_path.parent.parent / path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Legacy YesTiger rule pipeline: allin1/secondary analysis -> candidate slots -> draft callbook."
    )
    parser.add_argument("--audio", type=Path, help="Audio file. Required when --struct is omitted.")
    parser.add_argument("--struct", type=Path, help="Existing allin1 struct JSON. If omitted, allin1 is run first.")
    parser.add_argument("--library", type=Path, default=Path("knowledge/call_mix_library.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("ouen_analysis"))
    parser.add_argument("--allin1-device", default="cpu")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("LEGACY RULE PIPELINE")
    print("This wrapper is kept for old rule-based drafts. New work should prefer scripts/predict_tiny_pipeline.py.")

    struct_path = args.struct
    if struct_path is None:
        if args.audio is None:
            raise SystemExit("--audio is required when --struct is omitted.")
        audio_path = args.audio
        struct_path = Path("struct") / f"{audio_path.stem}.json"
        run_command(
            [
                str(root / "run_allin1.bat"),
                str(audio_path),
                "-o",
                "struct",
                "--no-multiprocess",
                "-d",
                args.allin1_device,
            ],
            cwd=root,
        )
    else:
        audio_path = args.audio or resolve_audio_from_struct(struct_path)

    if not struct_path.exists():
        raise SystemExit(f"Struct JSON not found: {struct_path}")
    if audio_path is None:
        raise SystemExit("--audio was not supplied and no audio path was found in the struct JSON.")

    run_command(
        [
            sys.executable,
            str(root / "scripts" / "secondary_audio_analysis.py"),
            "--struct",
            str(struct_path),
            "--audio",
            str(audio_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=root,
    )

    secondary_path = out_dir / f"{struct_path.stem}.secondary.json"
    run_command(
        [
            sys.executable,
            str(root / "scripts" / "candidate_slot_generator.py"),
            "--secondary",
            str(secondary_path),
            "--struct",
            str(struct_path),
            "--library",
            str(args.library),
            "--out-dir",
            str(out_dir),
            "--top-n",
            str(args.top_n),
        ],
        cwd=root,
    )

    candidates_path = out_dir / f"{struct_path.stem}.candidates.json"
    run_command(
        [
            sys.executable,
            str(root / "scripts" / "draft_callbook_generator.py"),
            "--candidates",
            str(candidates_path),
            "--out-dir",
            str(out_dir),
        ],
        cwd=root,
    )

    print("Legacy draft written:")
    print(out_dir / f"{struct_path.stem}.callbook.draft.json")
    print(out_dir / f"{struct_path.stem}.callbook.draft.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
