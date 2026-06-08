# allin1 Local Setup Notes

This project uses a local virtual environment at `.venv`.

Run allin1 through the wrapper so caches stay inside this project:

```powershell
.\run_allin1.bat .\songs\your_song.mp3 -o .\struct --no-multiprocess -d cpu
```

Useful checks:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\run_allin1.bat -h
```

Installed compatibility choices:

- `allin1==1.1.0`
- `madmom==0.16.1`
- `numpy==1.23.5`, because `madmom` is not compatible with NumPy 2.x
- `setuptools<81`, because `madmom` imports `pkg_resources`
- `natten==0.21.6`

The current NATTEN package no longer exposes the old functions expected by `allin1==1.1.0`, so the local package at
`.venv\Lib\site-packages\allin1\models\dinat.py` has a pure PyTorch fallback for those four functions.
