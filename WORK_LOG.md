# Work Log

Running log of in-progress work and open threads for this repo, meant to be
read at the start of a new session. Update it as things get resolved instead
of letting it go stale — remove/close items once done, don't just append.

## Active thread: Aug 5 conference paper rework

Full detail, findings, and next steps: **`HANDOVER_2026-07-31.md`** (read
this first; `HANDOVER_2026-07-26.md` still has the original design
rationale for the labeling-rework plan). Short version: reworking the
ground-truth labeling pipeline (schedule + unsupervised clustering on flow
features, replacing a raw SYN-count threshold) before retraining on the
local RTX 3070, so the paper can defensibly claim genuine visual detection
rather than "just counting TCP [SYN packets] and labeling based on that."

**2026-08-01 update**: Real-data pipeline complete and a second, bigger
workstream is in progress:

1. **Labeling rework — done, on real data.** All 5 CICDDoS2019 PCAP zips
   downloaded, unzipped, and passed through `extract_window_features.py`
   (137,285 window feature rows) and `cluster_validate_labels.py`
   (`dataset/window_labels_validated_v1.json`,
   `reports/cluster_validation_2026-08-01.md`). Found and fixed a real bug:
   the original script let a whole-day k=2 KMeans split pass its silhouette
   gate even when the "attack" cluster it picked wasn't actually syn-heavy
   (happened on `01-12-2018`, whose true SYN window is only ~5 minutes —
   the split was separating on unrelated traffic volume, not attack
   behavior). Added `attack_cluster_has_flood_signature()` — a second gate
   requiring the picked cluster's mean `syn_ratio` to exceed the other
   cluster's — with a new `schedule_priority_low_syn_purity` fallback mode
   and a regression test reproducing the exact failure. 28/28 tests passing
   after the fix (was 27/27 before this session). `03-11-2018` still
   cluster-validates cleanly (silhouette 0.64, corroborates the existing
   syn-count heuristic almost exactly); `01-12-2018` now correctly falls
   back to schedule-only labels instead of a spurious cluster result.

2. **Bigger pivot, in progress: new image encoding.** User-directed
   decision: the raw-byte-dump-reshaped-to-a-square image encoding
   (`pcap_to_images.py::raw_bytes_to_image_224x224`) is a bigger "why
   images, why CNN" reviewer risk than the labeling heuristic was — no
   header/payload structure, and the model (ResNeXt-50, trained from
   scratch) still uses ImageNet-style `RandomResizedCrop` +
   ImageNet mean/std, both natural-photo conventions that don't mean much
   for a byte dump. New: `scripts/pcap_to_flowpic_images.py` — adapts
   *FlowPic* (Shapira & Shavitt 2019): each 0.3s window becomes a 3-channel
   64x64 (packet-size × arrival-time) histogram (ch0=all packets,
   ch1=SYN-only, ch2=ACK|FIN|RST), log1p-scaled against a **fixed global**
   clip (not per-image min/max) so absolute traffic volume stays a real
   signal. 10/10 new unit tests passing. Two small additive CLI changes to
   `notebook/train_resnext.py`: `--norm-mean`/`--norm-std` (default
   unchanged = ImageNet stats; used here with dataset-computed stats
   instead) — `--no-augment` already existed and is reused (crop/flip don't
   make sense on a size×time histogram). Full ablation requested: build
   **both** the old byte-dump images and the new flowpic images from
   cluster-validated labels, retrain both, compare against the existing
   primary checkpoint.

3. **Blocker hit and resolved**: raw PCAPs had been deleted from both
   machines after step 1 (per the standing "clear space" instruction) —
   needed for image generation regardless of which encoding. Re-download
   blocked on an expired CIC dataset cookie; user re-exported a fresh one
   (Cookie-Editor, Netscape format), re-download completed in ~17 min this
   time. **Current blocker: server→local transfer is slow and link is
   sensitive to parallelism** — multiple parallel scp/rsync streams
   actively hurt throughput here (5-way parallel dropped combined speed to
   ~0.7MB/s, worse than one stream alone at ~1.3-2.5MB/s) and the SSH
   connection has dropped mid-transfer more than once with
   `message authentication code incorrect` (transient, not a real MITM/key
   issue — same key works fine on retry). Now running **sequential**
   (one file at a time) via a self-retrying loop
   (`/tmp/.../scratchpad/sequential_transfer.sh` — not in repo, scratch
   only). ETA for all 5 zips: ~3-4 hours from 2026-08-01 ~10:10 EEST given
   the observed ~1.5MB/s single-stream rate.

**Code is done and tested** (`scripts/pcap_to_flowpic_images.py`,
`scripts/generate_validated_images_batch.sh`,
`scripts/train_resnext_per_second_0p3_validated_v1.sh`,
`scripts/train_resnext_flowpic_v1.sh`,
`scripts/compute_flowpic_norm_stats.py`, the `cluster_validate_labels.py`
fix, the `train_resnext.py` CLI additions) — pushed to GitHub even though
the data pipeline (transfer → generate both image sets → split →
compute flowpic norm stats → retrain both → eval → comparison report) is
still running. See git log for the actual commit.

## Next steps (in order)

1. Sequential transfer of all 5 PCAP zips finishes (background, self-retrying,
   ETA ~3-4h from 2026-08-01 10:10 EEST).
2. Run `scripts/generate_validated_images_batch.sh <zip>` per zip (builds
   BOTH the byte-dump-validated_v1 and flowpic-validated_v1 image sets from
   the same unzip, then clears the zip+staging) — avoids a third
   re-download.
3. `scripts/split_train_val_test.py` on both new dataset roots.
4. `scripts/compute_flowpic_norm_stats.py` on the flowpic train split, plug
   the printed `NORM_MEAN`/`NORM_STD` into
   `scripts/train_resnext_flowpic_v1.sh` (currently has placeholder
   `0.5/0.25` values — must be replaced before that training run means
   anything).
5. Retrain both: `scripts/train_resnext_per_second_0p3_validated_v1.sh` then
   `scripts/train_resnext_flowpic_v1.sh` (sequential, one RTX 3070).
6. Eval both checkpoints via existing eval scripts
   (`HUAWEI_EVAL_ALLOW_NON_PRIMARY_CKPT=1`), write a 3-way comparison report
   (existing primary vs. validated_v1-bytedump vs. flowpic_v1) —
   accuracy/macro-F1/ddos-F1 side by side.
7. Still deferred to the end, per standing instruction: paper/report text
   rewrite (Section III.C fix, SYNShield spelling), Mohammed Farhat
   author/email issue, `install.sh` disposal, recovering the Overleaf
   `.tex` sources for the two loose PDFs at repo root.

## Earlier session — 2026-07-26 (untracked files at root)

### Untracked files at repo root (not yet committed)
- `install.sh` — **not part of this project.** This is Anthropic's own Claude
  Code CLI installer script, apparently downloaded/dropped into the repo root
  by accident. Should probably be deleted (not committed) unless the user
  says otherwise.
- `DDoS_Attack_Detection_Using_ResNeXt50_32x4d_with_MindSpore.pdf` — new
  28-page A4 revision of the BAU competition report. See below.
- `DDoS_Attack_Detection_Using_ResNeXt50_32x4d_with_MindSpore___paper.pdf` —
  new 5-page IEEE-style conference paper draft ("SYNShield"). See below.

### The two new PDFs — what was found
Both compiled today (~20:39–20:40) from LaTeX sources that are **not in this
repo** (no matching `.tex` produces either file; `research/main.tex` is
unchanged since 2026-04-03). They were likely edited/compiled in Overleaf and
the output PDFs dropped in here directly — the `.tex` sources need to be
pulled in separately if they should be tracked.

**1. Full report** (`..._with_MindSpore.pdf`, 28pp, A4)
- Same title/scope as the existing `research/DDoS_Attack_Detection_Using_ResNeXt50_32x4d_with_MindSpore.pdf`
  (2026-04-03), but a reworded revision:
  - Abstract has been rephrased (same reported numbers: 98.1% accuracy, 0.937
    macro F1, 0.3s-window model).
  - The "List of Abbreviations" page present in the April version is
    **missing** from this new one — looks like it dropped out, not
    intentionally cut.
- Authors on cover page: Mohamad Al Ghoush, Abed Al Rida Nehme, Wafik
  Ibrahim, Ahmad Sharkawi (ECE, Beirut Arab University, 2026).

**2. IEEE conference paper** (`..._paper.pdf`, 5pp, Letter, two-column)
- Titled internally "SYNShield" — a condensed conference-style writeup, not
  the thesis-style report.
- Author list **differs from the report**: Abed Al Rida Nehme, Mohamad Al
  Ghoush, **Mohammed Farhat** — drops Wafik Ibrahim, adds a name not seen
  elsewhere in the repo.
- Mohammed Farhat's email is an unfilled placeholder: `XXXXXXX@bau.edu.lb`.

### Open questions / likely next steps
- [ ] Find/recover the `.tex` source(s) for both new PDFs (Overleaf?) and
      decide whether to commit them into `research/` or `ict_innovation_overleaf/`.
- [ ] Confirm whether "Mohammed Farhat" is an actual added team member (fix
      placeholder email + decide if Wafik Ibrahim's removal from the paper
      author list is intentional) or a mistake.
- [ ] Decide if the missing abbreviations list in the report revision is
      intentional; restore if not.
- [ ] Decide fate of `install.sh` (delete vs. add to `.gitignore`).
- [ ] Nothing has been git-added/committed yet — all three untracked files
      are still sitting in the working tree as of this log entry.

## Prior state (from earlier exploration, still accurate as of 2026-07-26)
See `SYSTEM_DESCRIPTION.txt`, `depployemnt/PROGRESS.md`, and
`depployemnt/TODO.md` for the full project background (ResNeXt-50 SYN-flood
DDoS detector on Huawei MindSpore, live-deployed on a Huawei Cloud ECS
instance). Those docs are current and don't need restating here — this log
is only for things in flight that aren't captured elsewhere yet.
