# phototag (local folder: Orian)

Local face-tagging tool: scan a photo folder → label face clusters in a web
UI → export per-person folders. Single file `phototag.py`; see README.md for
usage. State in `phototag_data/`, face model in `models/` (auto-downloaded,
gitignored), venv in `.venv/` (Python 3.13 from /usr/local/bin/python3.13).

## Accounts & credentials

- GitHub: `razv2025/phototag` (private; shared with friends & family as
  collaborators). Push with the logged-in `gh` CLI.
- AWS: none — this project deliberately has no cloud footprint (photos are
  private; everything runs locally).

## Conventions

- Matching thresholds live as constants CONFIDENT/BORDERLINE in phototag.py
  and as `serve` flags; if real-corpus accuracy is off, tune flags first.
- The synthetic logic test (cluster → name → reclassify → reject → export)
  is not committed; it lives in past session scratchpads — recreate from git
  history of this file if needed.
