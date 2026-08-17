# phototag

Find the people you care about in a folder of hundreds of photos — fully
local, no cloud, nothing ever uploaded. Faces are detected and embedded with
InsightFace (ArcFace) on CPU; you name a few face clusters in a small web UI,
the tool tags everyone else automatically and learns from every
confirmation/rejection you make.

## Setup

Requires Python 3.10–3.13 and ~1 GB of disk (dependencies + face model).

```bash
git clone <this repo> && cd phototag
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

On Windows use `.venv\Scripts\pip` / `.venv\Scripts\python` instead of
`.venv/bin/...` below.

## Usage

```bash
# 1. Index all faces (JPG/PNG, recursive; cached — re-runs only touch new/changed photos)
.venv/bin/python phototag.py scan ~/path/to/photos

# 2. Label & review in the browser (opens automatically);
#    each person's name links to their own page with all their photos
.venv/bin/python phototag.py serve

# 3. (optional) Also build one folder per person on disk
.venv/bin/python phototag.py export ~/Desktop/tagged
# on Windows, or if you want real files instead of symlinks:
.venv/bin/python phototag.py export ~/Desktop/tagged --copy
```

The first `scan` downloads the face model (~300 MB) into `./models/` — one
time only, after that everything works offline.

## The web UI

- **Unidentified clusters** — groups of similar faces. Type a name on the
  clusters that are your targets; hit *Ignore* on everyone else. Naming a
  cluster makes those faces training exemplars and immediately re-classifies
  the whole corpus.
- **Review queue** — borderline matches. *Yes* confirms and adds the face as
  an exemplar; *No* records a rejection the matcher will never repeat for that
  face; you can also assign the face to a different person or ignore it.
  Every answer triggers a re-classification, so the queue shrinks as you go.
- **People** — per-person counts and faces (✓ = manually confirmed,
  % = auto-match score). The ✗ button un-tags a wrong auto-match, which also
  teaches the matcher. Click a person's name to open **their page**: every
  photo they appear in (confirmed + confident matches only, never unreviewed
  borderline ones), full-size on click.

## Tuning

Thresholds (cosine similarity to a person's exemplars):
`serve --confident 0.55 --borderline 0.38`. Raise `--confident` for fewer
false tags, lower `--borderline` to review more candidates.

All state lives in `./phototag_data/` (SQLite + face thumbnails). Delete that
folder to start over; your photos are never modified. Use `--data DIR` before
the subcommand to keep separate state per photo corpus.

Notes: only JPG/JPEG/PNG are scanned; faces smaller than ~36 px are skipped as
too small to identify reliably.

## Privacy

Everything runs on your machine: photos, face embeddings, and the labeling UI
(which binds to 127.0.0.1 only). Nothing is sent anywhere.
