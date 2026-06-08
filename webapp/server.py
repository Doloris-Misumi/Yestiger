import argparse
import cgi
import json
import mimetypes
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyzer import JOB_DIR, ROOT, UPLOAD_DIR, analyze_audio, load_example_result, save_analysis_result, slugify


STATIC_DIR = THIS_DIR / "static"


def safe_join(base: Path, raw_path: str) -> Path:
    requested = (base / raw_path.lstrip("/")).resolve()
    if base.resolve() not in requested.parents and requested != base.resolve():
        raise ValueError("Path escapes static root.")
    return requested


def guess_type(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "application/octet-stream"


def example_audio_path(song_id: str) -> Path:
    annotation_path = ROOT / "annotations" / song_id / f"{song_id}.annotation.json"
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    raw = (data.get("song") or {}).get("audio_path")
    if not raw:
        raise FileNotFoundError(f"No audio_path in {annotation_path}")
    path = Path(str(raw))
    audio = path if path.is_absolute() else ROOT / path
    if not audio.exists():
        raise FileNotFoundError(audio)
    return audio


class YesTigerHandler(BaseHTTPRequestHandler):
    server_version = "YesTigerWeb/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))

    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200) -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, download_name: str = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_json({"error": "file_not_found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", guess_type(path))
        self.send_header("Content-Length", str(path.stat().st_size))
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/" or path == "/index.html":
                self.send_file(STATIC_DIR / "index.html")
                return
            if path.startswith("/static/"):
                self.send_file(safe_join(STATIC_DIR, path[len("/static/") :]))
                return
            if path.startswith("/examples/"):
                self.send_file(safe_join(STATIC_DIR / "examples", path[len("/examples/") :]))
                return
            if path == "/api/songs":
                self.handle_songs()
                return
            if path.startswith("/api/examples/"):
                song_id = slugify(path.split("/")[-1])
                self.handle_example(song_id)
                return
            if path.startswith("/api/example-audio/"):
                song_id = slugify(path.split("/")[-1])
                self.send_file(example_audio_path(song_id))
                return
            if path.startswith("/api/jobs/"):
                self.handle_job_file(path)
                return
            self.send_json({"error": "not_found"}, status=404)
        except Exception as exc:
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self.send_json({"error": "not_found"}, status=404)
            return
        try:
            self.handle_analyze()
        except Exception as exc:
            self.send_json({"error": type(exc).__name__, "message": str(exc)}, status=500)

    def handle_songs(self) -> None:
        songs = []
        root = ROOT / "experiments" / "signal_callability"
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            song_id = directory.name
            result_path = directory / f"{song_id}.merged.loso_audio_vote_rf1_logreg1_gb1.barfit_action_call_spans.json"
            if not result_path.exists():
                continue
            title = song_id
            annotation_path = ROOT / "annotations" / song_id / f"{song_id}.annotation.json"
            if annotation_path.exists():
                data = json.loads(annotation_path.read_text(encoding="utf-8"))
                title = (data.get("song") or {}).get("title") or title
            songs.append({"song_id": song_id, "title": title})
        self.send_json({"songs": songs})

    def handle_example(self, song_id: str) -> None:
        result = load_example_result(song_id)
        if result.get("audio_path"):
            result["audio_url"] = f"/api/example-audio/{song_id}"
            result.pop("audio_path", None)
        result["downloads"] = {}
        self.send_json(result)

    def handle_job_file(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) != 4:
            self.send_json({"error": "bad_job_path"}, status=400)
            return
        _, _, job_id, filename = parts
        job_id = slugify(job_id)
        job_dir = JOB_DIR / job_id
        if filename == "result.json":
            self.send_file(job_dir / "result.json", download_name=f"{job_id}.result.json")
            return
        if filename == "callbook.md":
            self.send_file(job_dir / "callbook.md", download_name=f"{job_id}.callbook.md")
            return
        if filename == "audio":
            matches = list((UPLOAD_DIR / job_id).glob("*"))
            if not matches:
                self.send_json({"error": "audio_not_found"}, status=404)
                return
            self.send_file(matches[0])
            return
        self.send_json({"error": "bad_job_file"}, status=400)

    def handle_analyze(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_json({"error": "expected_multipart_form_data"}, status=400)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
        )
        if "audio" not in form:
            self.send_json({"error": "missing_audio"}, status=400)
            return
        field = form["audio"]
        if isinstance(field, list):
            field = field[0]
        original_filename = Path(field.filename or "uploaded_audio.wav").name
        original_path = Path(original_filename)
        suffix = original_path.suffix.lower()
        safe_suffix = suffix if suffix and len(suffix) <= 10 and all(char.isalnum() or char == "." for char in suffix) else ".audio"
        safe_stem = slugify(original_path.stem)[:80] or "uploaded_audio"
        filename = f"{safe_stem}{safe_suffix}"
        title = form.getfirst("title") or original_path.stem
        job_id = slugify(f"{safe_stem}_{slugify(title)}")[:40]
        if not job_id:
            job_id = "uploaded"
        job_id = f"{job_id}_{len(list(JOB_DIR.glob(job_id + '*'))):03d}"
        upload_dir = UPLOAD_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        audio_path = upload_dir / filename
        with audio_path.open("wb") as handle:
            shutil.copyfileobj(field.file, handle)

        result = analyze_audio(audio_path, title=title, job_id=job_id)
        result.setdefault("song", {})["original_audio_filename"] = original_filename
        result["audio_url"] = f"/api/jobs/{job_id}/audio"
        result["downloads"] = {
            "json": f"/api/jobs/{job_id}/result.json",
            "markdown": f"/api/jobs/{job_id}/callbook.md",
        }
        save_analysis_result(result, JOB_DIR / job_id)
        self.send_json(result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the YesTiger local web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    log_path = JOB_DIR.parent / "server-startup.log"
    server = ThreadingHTTPServer((args.host, args.port), YesTigerHandler)
    message = f"YesTiger web app listening on http://{args.host}:{args.port}"
    log_path.write_text(message + "\n", encoding="utf-8")
    print(message, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping YesTiger web app.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
