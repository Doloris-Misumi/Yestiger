# YesTiger Web Studio

Local web wrapper for uploading idol songs, estimating call sections, filling bar-fit support actions, previewing a synchronized action video, and exporting callbooks.

## Run

Install the optional MP3 fallback decoder if uploads fail with a `soundfile` decode error:

```powershell
.\.venv\Scripts\python.exe -m pip install miniaudio
```

```powershell
.\.venv\Scripts\python.exe webapp\server.py --host 127.0.0.1 --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

## Outputs

Uploaded audio and generated job files are saved under:

```text
webapp_runs/
```

The web app can export edited JSON, Markdown callbooks, and browser-recorded WebM videos.

For GitHub + Vercel deployment, see [DEPLOYMENT_VERCEL_ZH.md](../DEPLOYMENT_VERCEL_ZH.md).
