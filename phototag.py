#!/usr/bin/env python3
"""phototag — find target people in a folder of photos, fully locally.

Workflow:
  1. python phototag.py scan  ~/path/to/photos     # detect faces + embeddings (cached)
  2. python phototag.py serve                      # web UI: name clusters, review matches
  3. python phototag.py export ~/path/to/tagged    # symlink folders, one per person

All state lives in ./phototag_data (SQLite db + face thumbnails).
"""

import argparse
import io
import json
import os
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path

import numpy as np

# Cosine-similarity thresholds against a person's exemplar faces.
CONFIDENT = 0.55   # >= this: tagged automatically
BORDERLINE = 0.38  # >= this: sent to the review queue

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
THUMB_SIZE = 160
MIN_DET_SCORE = 0.50
MIN_FACE_PX = 36


# ---------------------------------------------------------------- storage

def open_db(data_dir: Path) -> sqlite3.Connection:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "thumbs").mkdir(exist_ok=True)
    db = sqlite3.connect(data_dir / "phototag.db", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS photos(
            id INTEGER PRIMARY KEY, path TEXT UNIQUE, mtime REAL);
        CREATE TABLE IF NOT EXISTS faces(
            id INTEGER PRIMARY KEY, photo_id INTEGER REFERENCES photos(id),
            emb BLOB, det_score REAL,
            person TEXT, source TEXT, score REAL,   -- source: manual|auto|borderline
            ignored INTEGER DEFAULT 0, cluster_id INTEGER);
        CREATE TABLE IF NOT EXISTS rejections(
            face_id INTEGER, person TEXT, UNIQUE(face_id, person));
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
    """)
    return db


def load_embeddings(db, face_ids):
    if not face_ids:
        return np.zeros((0, 512), dtype=np.float32)
    rows = db.execute(
        f"SELECT id, emb FROM faces WHERE id IN ({','.join('?' * len(face_ids))})",
        face_ids).fetchall()
    by_id = {r["id"]: np.frombuffer(r["emb"], dtype=np.float32) for r in rows}
    return np.stack([by_id[i] for i in face_ids])


# ---------------------------------------------------------------- scan

def load_face_app():
    from insightface.app import FaceAnalysis
    print("Loading face model (first run downloads ~300MB)...")
    # Keep the model inside the project folder instead of ~/.insightface
    app = FaceAnalysis(name="buffalo_l", root=str(Path(__file__).resolve().parent),
                       providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def sync_corpus(db, data_dir, root, get_app, verbose=False):
    """Bring the face index in line with the photo folder: drop deleted
    photos, (re)scan new or modified ones. Cheap no-op when nothing changed.
    Returns (changed_photos, new_faces)."""
    changed = 0
    for row in db.execute("SELECT id, path FROM photos").fetchall():
        if not Path(row["path"]).exists():
            db.execute("DELETE FROM faces WHERE photo_id=?", (row["id"],))
            db.execute("DELETE FROM photos WHERE id=?", (row["id"],))
            changed += 1
    known = {r["path"]: r["mtime"]
             for r in db.execute("SELECT path, mtime FROM photos")}
    todo = [p for p in sorted(root.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTS and p.is_file()
            and known.get(str(p)) != p.stat().st_mtime]
    new_faces = 0
    if todo:
        import cv2
        app = get_app()
        for i, path in enumerate(todo, 1):
            img = cv2.imread(str(path))
            if img is None:  # formats OpenCV can't decode (e.g. avif) — try Pillow
                try:
                    from PIL import Image
                    img = cv2.cvtColor(np.asarray(Image.open(path).convert("RGB")),
                                       cv2.COLOR_RGB2BGR)
                except Exception:
                    img = None
            if img is None:
                if verbose:
                    print(f"  ! unreadable, skipped: {path}")
                continue
            db.execute("DELETE FROM faces WHERE photo_id IN "
                       "(SELECT id FROM photos WHERE path=?)", (str(path),))
            db.execute("INSERT INTO photos(path, mtime) VALUES(?,?) "
                       "ON CONFLICT(path) DO UPDATE SET mtime=excluded.mtime",
                       (str(path), path.stat().st_mtime))
            photo_id = db.execute("SELECT id FROM photos WHERE path=?",
                                  (str(path),)).fetchone()["id"]
            for face in app.get(img):
                x1, y1, x2, y2 = face.bbox.astype(int)
                if (face.det_score < MIN_DET_SCORE
                        or min(x2 - x1, y2 - y1) < MIN_FACE_PX):
                    continue
                emb = face.normed_embedding.astype(np.float32)
                cur = db.execute(
                    "INSERT INTO faces(photo_id, emb, det_score) VALUES(?,?,?)",
                    (photo_id, emb.tobytes(), float(face.det_score)))
                save_thumb(data_dir, cur.lastrowid, img, (x1, y1, x2, y2))
                new_faces += 1
            if verbose:
                print(f"  [{i}/{len(todo)}] {path.name}")
    db.commit()
    return changed + len(todo), new_faces


def cmd_scan(args):
    root = Path(args.photos_dir).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"not a directory: {root}")
    data_dir = Path(args.data)
    db = open_db(data_dir)
    # Remember the folder so `serve` can auto-sync it on every page hit.
    db.execute("INSERT INTO meta VALUES('root',?) ON CONFLICT(key) "
               "DO UPDATE SET value=excluded.value", (str(root),))
    _, new_faces = sync_corpus(db, data_dir, root, load_face_app, verbose=True)

    reclassify(db)
    recluster(db)
    db.commit()
    n_faces = db.execute("SELECT COUNT(*) c FROM faces").fetchone()["c"]
    n_clusters = db.execute(
        "SELECT COUNT(DISTINCT cluster_id) c FROM faces WHERE cluster_id IS NOT NULL"
    ).fetchone()["c"]
    print(f"Done: {new_faces} new faces ({n_faces} total), "
          f"{n_clusters} unidentified clusters.")
    print("Next: python phototag.py serve")


def save_thumb(data_dir, face_id, img, bbox):
    import cv2
    x1, y1, x2, y2 = bbox
    # Pad the crop a bit for context.
    pw, ph = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
    h, w = img.shape[:2]
    crop = img[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]
    scale = THUMB_SIZE / max(crop.shape[:2])
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                             max(1, int(crop.shape[0] * scale))))
    cv2.imwrite(str(data_dir / "thumbs" / f"{face_id}.jpg"), crop)


# ---------------------------------------------------------------- matching

def reclassify(db, confident=CONFIDENT, borderline=BORDERLINE):
    """Re-score every non-manual face against all manual exemplars."""
    exemplars = db.execute(
        "SELECT id, person FROM faces WHERE source='manual' AND ignored=0").fetchall()
    db.execute("UPDATE faces SET person=NULL, source=NULL, score=NULL "
               "WHERE source IN ('auto','borderline')")
    if not exemplars:
        return
    persons = sorted({r["person"] for r in exemplars})
    ex_mat = load_embeddings(db, [r["id"] for r in exemplars])
    ex_person_idx = np.array([persons.index(r["person"]) for r in exemplars])

    cands = db.execute("SELECT id FROM faces WHERE ignored=0 "
                       "AND (source IS NULL OR source!='manual')").fetchall()
    cand_ids = [r["id"] for r in cands]
    if not cand_ids:
        return
    cand_mat = load_embeddings(db, cand_ids)
    sims = cand_mat @ ex_mat.T  # embeddings are L2-normalized

    rejected = {}
    for r in db.execute("SELECT face_id, person FROM rejections"):
        rejected.setdefault(r["face_id"], set()).add(r["person"])

    for row_idx, face_id in enumerate(cand_ids):
        best_person, best_sim = None, -1.0
        for p_idx, person in enumerate(persons):
            if person in rejected.get(face_id, ()):
                continue
            mask = ex_person_idx == p_idx
            sim = float(sims[row_idx][mask].max())
            if sim > best_sim:
                best_person, best_sim = person, sim
        if best_person is None or best_sim < borderline:
            continue
        source = "auto" if best_sim >= confident else "borderline"
        db.execute("UPDATE faces SET person=?, source=?, score=? WHERE id=?",
                   (best_person, source, best_sim, face_id))


def recluster(db):
    """Group the still-unidentified faces so the user can name them in bulk.

    Faces that match no group (DBSCAN noise) become singleton clusters so
    every detected face is visible and nameable in the UI."""
    from sklearn.cluster import DBSCAN
    db.execute("UPDATE faces SET cluster_id=NULL")
    rows = db.execute(
        "SELECT id FROM faces WHERE ignored=0 AND person IS NULL").fetchall()
    ids = [r["id"] for r in rows]
    if not ids:
        return
    if len(ids) == 1:
        labels = np.array([0])
    else:
        mat = load_embeddings(db, ids)
        labels = DBSCAN(eps=0.5, min_samples=2, metric="cosine").fit_predict(mat)
    next_id = int(labels.max()) + 1
    for face_id, label in zip(ids, labels):
        if label < 0:
            label, next_id = next_id, next_id + 1
        db.execute("UPDATE faces SET cluster_id=? WHERE id=?",
                   (int(label), face_id))


# ---------------------------------------------------------------- export

def export_links(db, out_dir: Path, copy=False):
    import shutil
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = db.execute("""
        SELECT DISTINCT f.person, p.path FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE f.ignored=0 AND f.person IS NOT NULL AND f.source IN ('manual','auto')
    """).fetchall()
    counts = {}
    for r in rows:
        person_dir = out_dir / r["person"]
        person_dir.mkdir(exist_ok=True)
        src = Path(r["path"]).resolve()
        link = person_dir / src.name
        n = 1
        if copy:
            while link.exists() and not (link.is_file() and
                                         link.stat().st_size == src.stat().st_size):
                link = person_dir / f"{src.stem}_{n}{src.suffix}"
                n += 1
            if not link.exists():
                shutil.copy2(src, link)
        else:
            while os.path.lexists(link) and not (
                    link.is_symlink() and os.readlink(link) == str(src)):
                link = person_dir / f"{src.stem}_{n}{src.suffix}"
                n += 1
            if not os.path.lexists(link):
                link.symlink_to(src)
        counts[r["person"]] = counts.get(r["person"], 0) + 1
    return counts


def cmd_export(args):
    db = open_db(Path(args.data))
    counts = export_links(db, Path(args.out_dir).expanduser().resolve(), copy=args.copy)
    for person, n in sorted(counts.items()):
        print(f"  {person}: {n} photos")
    if not counts:
        print("Nothing to export yet — run `serve` and label some people first.")


# ---------------------------------------------------------------- web UI

def build_state(db):
    persons = {}
    for r in db.execute("""
        SELECT person, source, COUNT(*) n, COUNT(DISTINCT photo_id) np
        FROM faces WHERE person IS NOT NULL AND ignored=0 AND source IN ('manual','auto')
        GROUP BY person, source"""):
        p = persons.setdefault(r["person"], {"name": r["person"], "manual": 0,
                                             "auto": 0, "photos": 0})
        p[r["source"]] = r["n"]
        p["photos"] += r["np"]

    review = [dict(r) for r in db.execute("""
        SELECT id, person, score FROM faces
        WHERE source='borderline' AND ignored=0 ORDER BY score DESC LIMIT 60""")]

    clusters = []
    for r in db.execute("""
        SELECT cluster_id, COUNT(*) n FROM faces
        WHERE cluster_id IS NOT NULL AND ignored=0 AND person IS NULL
        GROUP BY cluster_id ORDER BY n DESC LIMIT 200"""):
        samples = [x["id"] for x in db.execute(
            "SELECT id FROM faces WHERE cluster_id=? AND person IS NULL AND ignored=0 "
            "ORDER BY det_score DESC LIMIT 8", (r["cluster_id"],))]
        clusters.append({"id": r["cluster_id"], "size": r["n"], "samples": samples})

    totals = db.execute("SELECT COUNT(*) faces, COUNT(DISTINCT photo_id) photos "
                        "FROM faces").fetchone()
    return {"persons": sorted(persons.values(), key=lambda p: p["name"]),
            "review": review, "clusters": clusters, "totals": dict(totals)}


def cmd_serve(args):
    from flask import Flask, jsonify, request, send_file

    data_dir = Path(args.data)
    db = open_db(data_dir)
    lock = threading.Lock()
    flask_app = Flask(__name__)

    def refresh():
        reclassify(db, args.confident, args.borderline)
        recluster(db)
        db.commit()

    face_app = []  # lazy singleton — only loaded if new photos appear

    def get_app():
        if not face_app:
            face_app.append(load_face_app())
        return face_app[0]

    def autosync():
        """Pick up added/changed/removed photos on every data request."""
        row = db.execute("SELECT value FROM meta WHERE key='root'").fetchone()
        if not row or not Path(row["value"]).is_dir():
            return
        changed, _ = sync_corpus(db, data_dir, Path(row["value"]), get_app)
        if changed:
            refresh()

    @flask_app.get("/")
    def index():
        return PAGE

    @flask_app.get("/api/state")
    def state():
        with lock:
            autosync()
            return jsonify(build_state(db))

    @flask_app.get("/thumb/<int:face_id>")
    def thumb(face_id):
        return send_file(data_dir / "thumbs" / f"{face_id}.jpg")

    @flask_app.get("/photo/<int:face_id>")
    def photo(face_id):
        with lock:
            row = db.execute("SELECT p.path FROM faces f JOIN photos p "
                             "ON p.id=f.photo_id WHERE f.id=?",
                             (face_id,)).fetchone()
        return send_file(row["path"])

    @flask_app.get("/image/<int:photo_id>")
    def image(photo_id):
        with lock:  # the sqlite connection is shared across request threads
            row = db.execute("SELECT path FROM photos WHERE id=?",
                             (photo_id,)).fetchone()
        return send_file(row["path"])

    @flask_app.get("/api/search")
    def search():
        want = set(request.args.getlist("with"))
        avoid = set(request.args.getlist("without"))
        mode = request.args.get("exclusive")  # 'strict' | 'named' | None
        with lock:
            autosync()
            photos = {r["id"]: {"id": r["id"], "name": Path(r["path"]).name,
                                "persons": set()}
                      for r in db.execute("SELECT id, path FROM photos")}
            n_faces, n_want = {}, {}  # per photo: all detected faces / ✓-tagged
            for r in db.execute("SELECT photo_id, person, source, ignored FROM faces"):
                pid = r["photo_id"]
                n_faces[pid] = n_faces.get(pid, 0) + 1
                if (not r["ignored"] and r["person"]
                        and r["source"] in ("manual", "auto")):
                    photos[pid]["persons"].add(r["person"])
                    if r["person"] in want:
                        n_want[pid] = n_want.get(pid, 0) + 1

        def ok(p):
            if mode:  # exactly the ✓ people...
                if not want or p["persons"] != want:
                    return False
                if mode != "strict":  # 'named': unnamed extra faces are fine
                    return True
                # 'strict': every detected face must be one of the ✓ people
                return n_want.get(p["id"], 0) == n_faces.get(p["id"], 0)
            return want <= p["persons"] and not (avoid & p["persons"])

        out = sorted(({**p, "persons": sorted(p["persons"])}
                      for p in photos.values() if ok(p)),
                     key=lambda p: p["name"])
        return jsonify(out)

    @flask_app.get("/person/<name>")
    def person_page(name):
        import html as html_mod
        with lock:
            autosync()
            rows = db.execute("""
                SELECT f.id face_id, f.source, f.score, p.id photo_id, p.path
                FROM faces f JOIN photos p ON p.id=f.photo_id
                WHERE f.person=? AND f.ignored=0 AND f.source IN ('manual','auto')
                ORDER BY p.path""", (name,)).fetchall()
        photos = {}
        for r in rows:
            ph = photos.setdefault(r["photo_id"], {
                "path": r["path"], "face_id": r["face_id"],
                "confirmed": False, "score": 0.0})
            if r["source"] == "manual":
                ph["confirmed"] = True
            else:
                ph["score"] = max(ph["score"], r["score"] or 0.0)
        cards = []
        for ph in photos.values():
            fname = html_mod.escape(Path(ph["path"]).name)
            badge = ("✓ confirmed" if ph["confirmed"]
                     else f"{int(ph['score'] * 100)}% match")
            cards.append(
                f'<div class=card><a href="/photo/{ph["face_id"]}" target=_blank>'
                f'<img src="/photo/{ph["face_id"]}" loading=lazy></a>'
                f'<div class=cap title="{fname}">{fname}</div>'
                f'<span class=badge>{badge}</span></div>')
        return (PERSON_PAGE
                .replace("__NAME__", html_mod.escape(name))
                .replace("__COUNT__", str(len(photos)))
                .replace("__CARDS__", "".join(cards) or
                         '<p class=muted>no photos yet</p>'))

    @flask_app.get("/api/person/<name>")
    def person_faces(name):
        with lock:
            rows = db.execute(
                "SELECT id, source, score FROM faces WHERE person=? AND ignored=0 "
                "ORDER BY source='manual' DESC, score DESC LIMIT 200",
                (name,)).fetchall()
        return jsonify([dict(r) for r in rows])

    @flask_app.post("/api/rename")
    def rename():
        a = request.get_json()
        old, new = a.get("old", "").strip(), a.get("new", "").strip()
        if not old or not new:
            return jsonify({"error": "empty name"}), 400
        with lock:
            db.execute("UPDATE faces SET person=? WHERE person=?", (new, old))
            # merge rejection rows, dropping ones that would collide
            db.execute("UPDATE OR IGNORE rejections SET person=? WHERE person=?",
                       (new, old))
            db.execute("DELETE FROM rejections WHERE person=?", (old,))
            refresh()
        return jsonify({"ok": True})

    @flask_app.post("/api/action")
    def action():
        a = request.get_json()
        with lock:
            kind = a["type"]
            if kind == "name_cluster":  # all cluster faces become manual exemplars
                db.execute("UPDATE faces SET person=?, source='manual', score=NULL "
                           "WHERE cluster_id=? AND person IS NULL AND ignored=0",
                           (a["name"].strip(), a["cluster_id"]))
            elif kind == "ignore_cluster":
                db.execute("UPDATE faces SET ignored=1 "
                           "WHERE cluster_id=? AND person IS NULL", (a["cluster_id"],))
            elif kind == "confirm":     # borderline/auto -> manual exemplar
                db.execute("UPDATE faces SET source='manual', score=NULL "
                           "WHERE id=? AND person IS NOT NULL", (a["face_id"],))
            elif kind == "reject":      # "not this person" — remember and relearn
                row = db.execute("SELECT person FROM faces WHERE id=?",
                                 (a["face_id"],)).fetchone()
                if row and row["person"]:
                    db.execute("INSERT OR IGNORE INTO rejections VALUES(?,?)",
                               (a["face_id"], row["person"]))
                db.execute("UPDATE faces SET person=NULL, source=NULL, score=NULL "
                           "WHERE id=?", (a["face_id"],))
            elif kind == "assign":      # manually name one face
                db.execute("UPDATE faces SET person=?, source='manual', score=NULL, "
                           "ignored=0 WHERE id=?", (a["name"].strip(), a["face_id"]))
            elif kind == "ignore_face":
                db.execute("UPDATE faces SET person=NULL, source=NULL, ignored=1 "
                           "WHERE id=?", (a["face_id"],))
            refresh()
            return jsonify(build_state(db))

    url = f"http://127.0.0.1:{args.port}"
    with lock:
        refresh()
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"phototag UI on {url}  (Ctrl-C to stop)")
    flask_app.run(port=args.port, debug=False)


PERSON_PAGE = """<!doctype html><meta charset="utf-8"><title>__NAME__ — phototag</title>
<style>
 body{font:14px -apple-system,sans-serif;margin:0;background:#101418;color:#e8eaed}
 header{position:sticky;top:0;background:#1a2027;padding:10px 16px;display:flex;
        gap:16px;align-items:center;box-shadow:0 1px 4px #0008}
 h1{font-size:16px;margin:0} a{color:#7ab8f5;text-decoration:none} .muted{color:#889}
 section{padding:12px 16px;display:grid;gap:10px;
         grid-template-columns:repeat(auto-fill,minmax(240px,1fr))}
 .card{background:#1a2027;border-radius:10px;padding:8px;min-width:0}
 .card img{width:100%;height:180px;object-fit:cover;border-radius:6px;display:block}
 .cap{margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
      color:#9ab;font-size:12px}
 .badge{white-space:nowrap}
 .badge{font-size:11px;background:#2b3542;border-radius:4px;padding:1px 6px;margin-left:6px}
</style>
<header><a href="/">← all people</a>
 <h1 id=pname title="double-click to rename" style="cursor:text">__NAME__</h1>
 <span class=muted>__COUNT__ photos</span></header>
<section>__CARDS__</section>
<script>
const el=document.getElementById('pname'), orig=el.textContent;
el.ondblclick=()=>{el.contentEditable=true;el.focus();
 getSelection().selectAllChildren(el)};
el.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();el.blur()}
 else if(e.key==='Escape'){el.textContent=orig;el.blur()}};
el.onblur=async()=>{el.contentEditable=false;
 const n=el.textContent.trim();
 if(!n||n===orig){el.textContent=orig;return}
 const r=await fetch('/api/rename',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({old:orig,new:n})});
 if(r.ok)location.href='/person/'+encodeURIComponent(n);
 else el.textContent=orig};
</script>"""


PAGE = """<!doctype html><meta charset="utf-8"><title>phototag</title>
<style>
 body{font:14px -apple-system,sans-serif;margin:0;background:#101418;color:#e8eaed}
 header{position:sticky;top:0;background:#1a2027;padding:10px 16px;display:flex;
        gap:16px;align-items:center;box-shadow:0 1px 4px #0008;z-index:2}
 h1{font-size:16px;margin:0} h2{font-size:14px;margin:18px 0 8px;color:#9ab}
 section{padding:4px 16px 12px} .muted{color:#889}
 .card{display:inline-block;vertical-align:top;background:#1a2027;border-radius:10px;
       padding:8px;margin:5px;max-width:360px}
 .thumbs img{height:76px;border-radius:6px;margin:2px;cursor:pointer}
 .review .thumbs img{height:110px}
 button{background:#2b3542;color:#e8eaed;border:0;border-radius:6px;padding:5px 10px;
        margin:2px;cursor:pointer} button:hover{background:#3a4756}
 button.ok{background:#1e5c3a} button.no{background:#5c2626}
 input,select{background:#0d1116;color:#e8eaed;border:1px solid #333;border-radius:6px;
        padding:5px 8px} .score{color:#8fb573;font-weight:600}
 .badge{font-size:11px;background:#2b3542;border-radius:4px;padding:1px 6px;margin-left:6px}
 a.plink{color:#7ab8f5;text-decoration:none} a.plink:hover{text-decoration:underline}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-top:8px}
 .pcard{background:#1a2027;border-radius:10px;padding:8px;min-width:0}
 .pcard img{width:100%;height:140px;object-fit:cover;border-radius:6px;cursor:pointer;display:block}
 .fname{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin:5px 0 3px;color:#9ab;font-size:12px}
 .pcard .badge{margin:0 4px 3px 0;display:inline-block}
</style>
<header><h1>phototag</h1><span id=stats class=muted></span></header>
<section id=search></section>
<section id=people></section>
<section id=review class=review></section>
<section id=clusters></section>
<script>
let S=null, openPerson=null, filt={}, exclusive='';
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function act(a){S=await (await fetch('/api/action',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(a)})).json();
  render()}
async function refresh(){S=await (await fetch('/api/state')).json();render()}
function img(id){return `<img src="/thumb/${id}" onclick="window.open('/photo/${id}')" title="click: full photo">`}
function personOptions(){return S.persons.map(p=>`<option>${esc(p.name)}</option>`).join('')}
function render(){
 const y=scrollY;
 $('stats').textContent=`${S.totals.photos} photos · ${S.totals.faces} faces`;
 renderSearch(); doSearch();
 $('people').innerHTML='<h2>People</h2>'+(S.persons.length?'':'<span class=muted>none yet — name a cluster below</span>')+
  S.persons.map(p=>`<div class=card><b><a class=plink href="/person/${encodeURIComponent(p.name)}" title="open ${esc(p.name)}'s page">${esc(p.name)}</a></b>
   <span class=badge>${p.manual} confirmed</span><span class=badge>${p.auto} auto</span>
   <button onclick="togglePerson('${esc(p.name)}')">${openPerson===p.name?'hide':'show faces'}</button>
   <div id="pf-${esc(p.name)}"></div></div>`).join('');
 if(openPerson)loadPerson(openPerson);
 $('review').innerHTML='<h2>Review queue — is this them?</h2>'+(S.review.length?'':'<span class=muted>empty 🎉</span>')+
  S.review.map(f=>`<div class=card><div class=thumbs>${img(f.id)}</div>
   <div><b>${esc(f.person)}</b>? <span class=score>${(f.score*100|0)}%</span></div>
   <button class=ok onclick='act({type:"confirm",face_id:${f.id}})'>✓ Yes</button>
   <button class=no onclick='act({type:"reject",face_id:${f.id}})'>✗ No</button>
   <select id=as-${f.id}><option value="">someone else…</option>${personOptions()}</select>
   <button onclick='assignSel(${f.id})'>set</button>
   <button onclick='act({type:"ignore_face",face_id:${f.id}})'>ignore</button></div>`).join('');
 $('clusters').innerHTML='<h2>Unnamed faces (most occurrences first) — name to get a personal page, or ignore</h2>'+
  (S.clusters.length?'':'<span class=muted>none</span>')+
  S.clusters.map(c=>`<div class=card><div class=thumbs>${c.samples.map(img).join('')}</div>
   <div>${c.size} face${c.size>1?'s':''}</div>
   <input id=cn-${c.id} placeholder="person name">
   <button class=ok onclick='nameCluster(${c.id})'>Name</button>
   <button onclick='act({type:"ignore_cluster",cluster_id:${c.id}})'>Ignore</button></div>`).join('');
 scrollTo(0,y);
}
function nameCluster(id){const v=$('cn-'+id).value.trim();if(v)act({type:'name_cluster',cluster_id:id,name:v})}
function assignSel(id){const v=$('as-'+id).value;if(v)act({type:'assign',face_id:id,name:v})}
function togglePerson(name){openPerson=openPerson===name?null:name;render()}
async function loadPerson(name){const faces=await (await fetch('/api/person/'+encodeURIComponent(name))).json();
 const el=document.getElementById('pf-'+name);if(!el)return;
 el.innerHTML='<div class=thumbs>'+faces.map(f=>`<span style="display:inline-block;text-align:center">
  ${img(f.id)}<br><span class=muted>${f.source==='manual'?'✓':(f.score*100|0)+'%'}</span>
  <button class=no title="not them" onclick='act({type:"reject",face_id:${f.id}})'>✗</button></span>`).join('')+'</div>'}
function cycle(name){
 filt[name]=exclusive?(filt[name]==='with'?undefined:'with')
  :(filt[name]==='with'?'without':filt[name]==='without'?undefined:'with');
 if(!filt[name])delete filt[name];renderSearch();doSearch()}
function exclChange(m,v){exclusive=v?m:'';
 if(exclusive)for(const n of Object.keys(filt))if(filt[n]==='without')delete filt[n];
 renderSearch();doSearch()}
function renderSearch(){
 for(const n of Object.keys(filt))if(!S.persons.some(p=>p.name===n))delete filt[n];
 $('search').innerHTML=`<h2>Search photos — click names to cycle: ✓ must appear${exclusive?'':' → ✗ must not'} → off</h2>`+
  (S.persons.length?
   `<label title="nobody else is on the photo at all — even unnamed faces"><input type=checkbox
     ${exclusive==='strict'?'checked':''} onchange="exclChange('strict',this.checked)"> only them </label>
    <label title="no other named person; unnamed faces may still appear"><input type=checkbox
     ${exclusive==='named'?'checked':''} onchange="exclChange('named',this.checked)"> only these named </label>`+
   S.persons.map(p=>{const st=filt[p.name];
   return `<button class="${st==='with'?'ok':st==='without'?'no':''}"
    onclick='cycle(${esc(JSON.stringify(p.name))})'>${st==='with'?'✓ ':st==='without'?'✗ ':''}${esc(p.name)}</button>`}).join('')
   +'<div id=sres></div>':'<span class=muted>name someone below first</span>')}
async function doSearch(){if(!S.persons.length)return;
 const q=new URLSearchParams();
 for(const[n,s]of Object.entries(filt))q.append(s==='with'?'with':'without',n);
 if(exclusive)q.append('exclusive',exclusive);
 const res=await (await fetch('/api/search?'+q)).json();
 const el=$('sres');if(!el)return;
 const y=scrollY;
 el.innerHTML=`<div class=muted style="margin:6px 0">${res.length} matching photos</div><div class=grid>`+
  res.map(ph=>`<div class=pcard><img src="/image/${ph.id}" loading=lazy onclick="window.open('/image/${ph.id}')">
   <div class=fname title="${esc(ph.name)}">${esc(ph.name)}</div>
   <div>${ph.persons.map(n=>`<span class=badge>${esc(n)}</span>`).join('')||'<span class=muted>nobody tagged</span>'}</div></div>`).join('')+'</div>';
 scrollTo(0,y)}
refresh();
</script>"""


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="phototag_data",
                    help="state directory (default: ./phototag_data)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("scan", help="index faces in a photo folder")
    p.add_argument("photos_dir")
    p.set_defaults(fn=cmd_scan)

    p = sub.add_parser("serve", help="open the labeling/review web UI")
    p.add_argument("--port", type=int, default=5088)
    p.add_argument("--confident", type=float, default=CONFIDENT)
    p.add_argument("--borderline", type=float, default=BORDERLINE)
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("export", help="build symlink folders per person")
    p.add_argument("out_dir")
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of symlinking (e.g. on Windows)")
    p.set_defaults(fn=cmd_export)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
