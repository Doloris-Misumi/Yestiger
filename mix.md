# PROJECT: YesTiger
## Context-aware MIX / Callbook Generation System

Goal:

Build a system that automatically generates idol/anison/seiyuu live MIX books (callbooks) from music.

Input:
- song audio
- optional lyrics
- optional metadata (idol group / seiyuu / franchise)

Output:
- timestamped MIX/call script
- intensity curve
- live-style-specific recommendation

---

# CORE IDEA

This is NOT:

detect BPM -> insert random MIX

This IS:

music structure
+ lyric semantics
+ emotional progression
+ otaku live culture
+ performance style

to decide:

- where to hype
- where to stay silent
- where to insert MIX
- where to clap
- where to do Fuwa
- where to do name call

The system acts like:

AI concert/ouen director

---

# IMPORTANT DESIGN DECISION

DO NOT let LLM directly process raw audio.

Correct architecture:

Audio
 -> MIR analyzer
 -> structured music representation
 -> candidate slot generator
 -> LLM arranger
 -> callbook generator

LLM should focus on:

high-level emotional arrangement

NOT:

beat detection

---

# SYSTEM ARCHITECTURE

```text
[Audio File]
      |
      v
+------------------+
| MIR Analyzer     |
| BPM / beats      |
| sections         |
| energy curve     |
+------------------+
      |
      v
+------------------+
| Lyrics Analyzer  |
| emotion          |
| vocal density    |
| keywords         |
+------------------+
      |
      v
+------------------+
| Candidate Slot   |
| Generator        |
+------------------+
      |
      v
+------------------+
| LLM Arranger     |
| emotional logic  |
| MIX selection    |
| density control  |
+------------------+
      |
      v
+------------------+
| Callbook Output  |
+------------------+