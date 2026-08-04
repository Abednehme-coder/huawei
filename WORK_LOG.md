# Work Log

Running log of in-progress work and open threads for this repo, meant to be
read at the start of a new session. Update it as things get resolved instead
of letting it go stale — remove/close items once done, don't just append.

## Active thread: Aug 5 conference paper rework

Full detail, findings, and next steps: **`HANDOVER_2026-08-04.md`** (read
this first — supersedes `HANDOVER_2026-08-03.md`, which supersedes
`HANDOVER_2026-07-31.md`; `HANDOVER_2026-07-26.md` still has the original
design rationale for the labeling-rework plan).
Short version: reworking the
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

**2026-08-04 update, overnight unattended run — DONE, all 5 zips + both image
sets complete and verified.** User asked to make sure all 5 raw PCAP zips are
present and ready. Turned up two things while checking:

1. Feature/label extraction (`window_features_real.csv`, 137,285 rows) was
   already done for all 5 zips — confirmed via `dataset/extract_*.log`
   (cumulative row counts chain correctly across all 5).
2. Image generation was only done for 4/5 zips (no `dataset/gen_0250-0499.log`)
   — expected. Auditing those 4 found the byte-dump set short by ~21,062
   files vs. what the generation logs claimed to have saved (flowpic matched
   its own logs exactly). Root cause found after regenerating: it was **all**
   in `PCAP-03-11.zip`'s original run — that zip's original byte-dump pass
   logged 21,062 saved images but almost none of them actually persisted to
   disk, while its flowpic counterpart wrote fine, and the *other* 3 zips'
   original byte-dump output was already correct. Not root-caused *why* that
   one run lost its writes, but re-running the generator against the
   re-fetched zip fixed it cleanly.

4 of the 5 raw zips had been deleted locally after processing (old "clear
space" policy, before the 2026-08-03 policy change — see
`HANDOVER_2026-08-03.md` §2), so fixing the byte-dump gap required
re-fetching those zips too, not just finishing the 5th. Built and ran an
unattended pipeline overnight to do this:

- `dataset/PCAPs_raw/parallel_fetch_generic.sh` — resumable parallel-chunk
  fetcher generalized from `parallel_fetch_0250.sh` (32MiB-chunk/8-way SSH
  `dd`), parameterized by zip name + target size, always on a `START_FROM=0`
  grid (the original 0250-0499 run had a grid-alignment resume bug from a
  stale nonzero `START_FROM` that silently threw away resume credit for its
  first ~450MB — not worth going back to fix, just noted so it isn't
  mistaken for zip corruption if noticed later).
- `dataset/PCAPs_raw/master_pipeline.sh` — orchestrated all 5 sequentially
  (waited for the already-running zip-1 fetch rather than double-transferring,
  then the remaining 4 smallest-first), verifying each with `unzip -t` before
  running `scripts/generate_validated_images_batch.sh`. Ran end-to-end in
  ~5h10m (00:16–05:29 EEST), fully self-healing, no manual intervention.

**Final verified state**: all 5 raw zips present in `dataset/PCAPs_raw/` at
their correct server-side sizes. Both `images_per_second_window_0p3_validated_v1`
and `images_flowpic_0p3_validated_v1` now contain exactly 96,770 PNGs each —
matching the full `window_labels_validated_v1.json` label count exactly (no
gap, no shortfall, both encodings in lockstep). Ready for the split step.

**2026-08-04, later same session — scope pivot: supervised-vs-unsupervised
comparison is now the paper's main new angle.** User-directed decision after
discussion: instead of just finishing the planned bytedump-vs-flowpic
encoding ablation, add a genuinely unsupervised baseline (trained with no
ddos labels) and compare it against the supervised ResNeXt-50 classifier on
the *same* held-out test set, on the **flowpic encoding only** (chosen for
speed — 64x64x3 vs 224x224 trains much faster on the one RTX 3070 — bytedump
supervised training is deprioritized, not cancelled, revisit if time allows).
Deadline (2026-08-05) has some flexibility per user, so doing this properly
rather than rushing it.

- **Split methodology decision**: rejected `split_train_val_test.py`'s
  default (random per-image, stratified only by class) in favor of a new
  `scripts/split_flowpic_blocked.py` — splits by whole source-pcap file
  ("stem"), never by individual 0.3s window, because adjacent windows in a
  capture are highly correlated and a random split risks near-duplicate
  windows leaking across train/test, which would inflate eval numbers and
  undercut the exact "genuine visual detection, not a labeling artifact"
  claim this whole labeling-rework effort exists for.
- Found the 11 ddos-bearing pcap stems (out of 965 total) fall into exactly
  two contiguous blocks matching CICDDoS2019's two separate attack days
  (`01-12-2018`: 3 stems/966 ddos windows; `03-11-2018`: 8 stems/1567 ddos
  windows). Used this directly: **test = the entire `01-12-2018` attack day,
  held out whole** — a real cross-day generalization test (unseen attack
  execution, not just unseen windows from an attack the model saw part of).
  Val = trailing 2 stems of the `03-11-2018` block (165 ddos, for
  early-stopping/threshold calibration). Train = remaining 6 attack stems
  (1402 ddos). Normal-only stems (954 of them) bucketed to round out val/test
  normal counts to ~10% each — had to fix a bug in the first version where
  greedy-filling from a *randomly shuffled* stem order let one huge outlier
  stem (some stems are ~25,210 windows vs. a median of 19 — "first sub-pcap
  in a zip" stems cover much longer spans) blow the val target 3.6x over;
  fixed by filling smallest-stems-first so many small stems hit the target
  precisely instead of one big one overshooting it.
- **Final split** (verified on disk):
  train: ddos=1402, normal=75376 (173 stems) | val: ddos=165, normal=9438
  (331 stems) | test: ddos=966, normal=9423 (461 stems). Normal hits ~80/10/10
  almost exactly; ddos is intentionally *not* proportional (55/6.5/38%)
  because test deliberately holds the whole cross-day attack block, not a
  proportional sample — that's the point of the design, not an imbalance bug.
- Still in progress: norm stats + supervised retrain on this new split,
  designing+training a MindSpore conv autoencoder (train on normal-only
  windows, no ddos labels seen), threshold calibration on val, final
  comparison on the shared test set. See task list in-session for live state
  if this gets interrupted — tasks 2-5 as of this writing.

**2026-08-04, resumed after unplanned power loss.** Machine turned off
mid-run: the flowpic autoencoder had gotten through epoch 1
(`val_mse=0.002346`, 296s) and died mid-write of epoch 2's checkpoint
(`autoencoder_best.ckpt` left as a 0-byte file); the supervised flowpic
retrain (`train_resnext_flowpic_v1.sh`) had not produced any output yet.
Resumed: cleared the corrupted 0-byte checkpoint and crashed logs (kept as
`*.crashed_2026-08-04` for reference), computed real flowpic norm stats
(`NORM_MEAN="0.0071 0.0003 0.0004"` `NORM_STD="0.0592 0.0157 0.0145"`, over
76,778 train images — sparse histograms, hence the small values) and baked
them into `train_resnext_flowpic_v1.sh` as the new default (was a
placeholder 0.5/0.25). Blocked flowpic split (`split_flowpic_blocked.py`)
was already done pre-crash and verified correct (train/val/test counts match
exactly). The bytedump validated_v1 split has NOT been run yet (its `train/`
still holds all 96,770 images, unsplit) — matches the deprioritization
decision above, not a bug.

Restarted both jobs. First attempt at the autoencoder retrain failed
immediately — invoked `train_flowpic_autoencoder.py`/
`eval_flowpic_autoencoder.py` directly without the `CUDA_HOME`/
`LD_LIBRARY_PATH` env that the `.sh` wrappers set, so MindSpore couldn't find
GPU libs and errored on `device_target=GPU`. The supervised retrain (run via
its `.sh` wrapper, which does set that env) is fine and running normally.
Re-running the autoencoder job with the correct env, sequenced after the
supervised job finishes (single RTX 3070, one job at a time, per the
existing convention here).

**2026-08-04, second unplanned power loss, ~14:00 EEST.** Machine shut down
again mid-run: the supervised flowpic retrain had gotten through epochs 1-4
cleanly (checkpoints saved 11:40-12:00, best-so-far at 11:54) and died
mid-epoch-5 (step 744/2399, garbled last log line from the mid-write cut).
The autoencoder job never got its restart in this window — the two log
files sitting in `model/flowpic_autoencoder_v1/` from ~10:47-10:48 were
leftover **failed** attempts (ran before the env fix, same
`device_target=GPU`/missing `CUDA_HOME` bug as before) that hadn't been
cleaned up yet when the second crash hit; renamed to
`*.crashed_2026-08-04_1047_gpuenv` / `*_1048_gpuenv` for the record.

Resumed again: no resume/checkpoint-load path exists in
`notebook/train_resnext.py`'s training loop (there's a `load_ckpt()` helper
but it's for loading a pretrained backbone, not mid-run optimizer/LR-schedule
state), so the supervised retrain restarts from scratch — the epoch 1-4
progress from this crash is lost (~30min of compute, not data). Tried to
clear the stale epoch-1-4 `.ckpt` files from `model/flowpic_0p3_validated_v1/`
before relaunching but that `rm` was blocked by the sandbox permission
classifier; left in place — harmless, `ModelCheckpoint`/best-ckpt tracking
in the new run overwrites them by filename as it reaches each epoch, and
the eval scripts always take an explicit `--ckpt` path so a stale file
can't silently get picked up.

Relaunched both jobs sequentially (supervised retrain → autoencoder retrain
→ autoencoder eval) as one `nohup ... & disown` background chain so they
survive this session ending — **does not** survive another actual machine
power-off, only terminal/session death. Two power losses in one day on this
machine is worth a look if it happens a third time (thermal? PSU? accidental
sleep-on-lid-close?) — not investigated, out of scope for this session.
Launched ~14:05 EEST. Progress:
```
tail -f model/flowpic_0p3_validated_v1/stdout.log       # supervised retrain
tail -f model/flowpic_autoencoder_v1/stdout.log          # autoencoder (starts after the above finishes)
ps aux | grep "[t]rain_resnext.py\|[t]rain_flowpic_autoencoder"
```

**2026-08-04, THIRD power loss, ~14:09 EEST (new session resuming work).**
Machine was found freshly rebooted (`uptime` showed 5 min at session start,
~14:14) — the 14:05 chain above died before training even started (log cuts
off right after the MindSpore context-init warnings, no epoch progress, no
new checkpoint files written past the pre-crash 11:40-12:00 ones). User
confirmed this (and presumably the two earlier ones today) is a genuine mains
power outage, not a machine fault — no thermal/PSU/lid-sleep investigation
needed.

Relaunch also hit a new, unrelated snag: `scripts/train_resnext_flowpic_v1.sh`
had lost its executable bit (`-rw-rw-r--`, shows as locally modified in `git
status` — likely an artifact of the unclean shutdown, not an intentional
edit). Fixed with `chmod +x`; first relaunch attempt failed fast with
`Permission denied` before this fix (harmless — the `&&`-chained autoencoder
stage never ran, no partial state to clean up).

Relaunched the same three-stage chain (supervised retrain → autoencoder
retrain → autoencoder eval) again via `nohup bash -c '... && ... && ...' &
disown`, confirmed alive past context-init this time (PID running,
`train_resnext.py` GPU process visible in `ps`, correct args including the
baked-in norm stats). `TO_RUN.sh` updated to reflect this restart. Same
caveat as before: survives terminal death, not another power-off.

**Status as of 16:39 EEST (this session, still in progress)**: stage 1
(supervised flowpic retrain) at epoch 19/40, best checkpoint so far epoch 16
(`f1_ddos=0.756`, `acc=0.989`), ~6:50/epoch stable. GPU well-utilized (97%
SM, checked via `nvidia-smi`) — not a bottleneck. Estimated ~48min-2h23min
left on this stage depending on early-stopping vs. running to completion,
then autoencoder stage (~1.25-4.1h depending on its own early stopping),
then a fast eval stage. Full detail, ETA math, and the exact relaunch
command for a hypothetical 4th outage: **`HANDOVER_2026-08-04.md`**.

## Next steps (in order)

1. ~~Sequential transfer of all 5 PCAP zips finishes~~ — **done 2026-08-04
   05:29 EEST**, see above. All 5 zips present + verified, both image sets
   complete (96,770 PNGs each, matching label ground truth exactly).
2. ~~Run `scripts/generate_validated_images_batch.sh <zip>` per zip~~ — done
   for all 5 as part of the above.
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
