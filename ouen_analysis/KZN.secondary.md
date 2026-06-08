# Secondary Audio Analysis

This is an audio-evidence refinement layer over allin1. Times are candidates for human verification.

## Refined Segments

| Start | End | Parent | Refined Label | Confidence | Reason |
|---:|---:|---|---|---:|---|
| 00:00.00 | 00:10.73 | intro | intro | 0.45 | No strong internal boundary detected; kept allin1 label: intro. |
| 00:10.73 | 00:31.04 | verse | verse_subsection | 0.47 | Internal timbre/energy change inside allin1 verse; check as A-melo repeat or smaller verse unit. |
| 00:31.04 | 00:46.28 | verse | pre_chorus_build_candidate | 0.55 | Last verse piece before chorus with rising energy; check as ietora/pre-chorus candidate. |
| 00:46.28 | 01:06.59 | chorus | chorus | 0.45 | No strong internal boundary detected; kept allin1 label: chorus. |
| 01:06.59 | 01:26.92 | chorus | chorus | 0.45 | No strong internal boundary detected; kept allin1 label: chorus. |
| 01:26.92 | 01:37.07 | verse | post_chorus_interlude_candidate | 0.37 | Early verse piece after chorus; check as post-chorus interlude/tiger-fire candidate. |
| 01:37.07 | 01:57.40 | verse | verse_subsection | 0.33 | Internal timbre/energy change inside allin1 verse; check as A-melo repeat or smaller verse unit. |
| 01:57.40 | 02:12.62 | verse | pre_chorus_candidate | 0.24 | Last verse piece before chorus; check as B-melo/pre-chorus. |
| 02:12.62 | 02:32.92 | chorus | chorus | 0.45 | No strong internal boundary detected; kept allin1 label: chorus. |
| 02:32.92 | 02:55.80 | chorus | chorus | 0.45 | No strong internal boundary detected; kept allin1 label: chorus. |
| 02:55.80 | 03:23.73 | inst | inst | 0.45 | No strong internal boundary detected; kept allin1 label: inst. |
| 03:23.73 | 03:36.43 | solo | solo | 0.45 | No strong internal boundary detected; kept allin1 label: solo. |
| 03:36.43 | 03:49.14 | verse | verse_subsection | 0.28 | Internal timbre/energy change inside allin1 verse; check as A-melo repeat or smaller verse unit. |
| 03:49.14 | 04:05.64 | verse | pre_chorus_candidate | 0.28 | Last verse piece before chorus; check as B-melo/pre-chorus. |
| 04:05.64 | 04:28.48 | chorus | chorus | 0.45 | No strong internal boundary detected; kept allin1 label: chorus. |
| 04:28.48 | 04:53.90 | chorus | chorus | 0.45 | No strong internal boundary detected; kept allin1 label: chorus. |
| 04:53.90 | 05:04.08 | outro | outro | 0.48 | Kept allin1 label: outro. |
| 05:04.08 | 05:10.73 | outro | outro | 0.48 | Kept allin1 label: outro. |
| 05:10.73 | 05:15.40 | end | end | 0.55 | Short segment; kept allin1 label: end. |

## Candidate Boundaries

| Time | Parent | Score | Novelty | Timbre | Energy | Onset |
|---:|---|---:|---:|---:|---:|---:|
| 00:31.04 | verse 00:10.73-00:46.28 | 0.47 | 0.66 | 0.44 | 0.27 | 0.14 |
| 01:37.07 | verse 01:26.92-02:12.62 | 0.33 | 0.47 | 0.35 | 0.09 | 0.09 |
| 01:57.40 | verse 01:26.92-02:12.62 | 0.24 | 0.42 | 0.19 | 0.05 | 0.01 |
| 03:49.14 | verse 03:36.43-04:05.64 | 0.28 | 0.45 | 0.23 | 0.09 | 0.05 |
| 05:04.08 | outro 04:53.90-05:10.73 | 0.64 | 0.74 | 0.75 | 0.44 | 0.14 |