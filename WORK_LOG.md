# Work Log

Running log of in-progress work and open threads for this repo, meant to be
read at the start of a new session. Update it as things get resolved instead
of letting it go stale — remove/close items once done, don't just append.

## Active thread: Aug 5-6 conference paper rework — CLOSED OUT 2026-08-06

**Read `HANDOVER_2026-08-06.md` first** — supersedes `HANDOVER_2026-08-05.md`
(which is now stale on multiple points, not just the val→test-gap number).
This thread's findings are now written into the actual paper sources
(`research/main.tex`, `research/synshield_paper/paper.tex`) and a standalone
report (`reports/label_validation_and_generalization_gap_2026-08-06.md`),
committed and pushed — this is no longer just log entries, it's landed in
the deliverables. Short version of the overall project: reworked the
ground-truth labeling pipeline (schedule + unsupervised clustering on flow
features, replacing a raw SYN-count threshold), retrained on a stricter
cross-day held-out split, and found + reported a real (if partially
label-driven) generalization gap rather than either hiding it or
overclaiming a fix. See the 2026-08-05 ~23:35 EEST entry onward, below, for
the final session's blow-by-blow (scope-cut decision, push-access fix,
recovered Overleaf sources, paper edits).

**2026-08-05 ~19:57 EEST update — official-label CSV finally in hand, via a
third server.** Resuming this session: the local `CSV-01-12.zip` download
(started 2026-08-04 23:24 EEST direct from cicresearch.ca) had died
sometime overnight — process gone, file stuck at 336MB, fails zip-integrity
check. User had *also* tried fetching it on the `root@91.99.170.219`
staging server (per the 2026-08-05 handover, that IP is blocked by
cicresearch.ca specifically for this endpoint) — checked it, confirmed:
`fetch.log` there shows only one stalled attempt, no zip file anywhere on
disk, no process running. **User then pointed at a third machine,
`abed@62.238.30.120`** (password auth), which turned out to already have
both `dataset/cicddos2019_csvs/CSV-01-12.zip` (2.33GB, contains
`Syn.csv` + 10 other attack-type CSVs for that day) and `CSV-03-11.zip`
(919MB) fully downloaded and zip-valid. No `sshpass`/`expect` available
locally and no `sudo` to install one — used `pip3 install --user paramiko`
instead. Pulling both files now via a paramiko SFTP script
(`dataset/CIC_official_labels/pull_from_62.238.30.120.log` for progress),
replacing the corrupt local `CSV-01-12.zip`. Transfer rate is modest
(~0.7 MB/s observed early on) but this is a one-time pull of files that
already exist in full elsewhere, not a live restart-prone download like
the cicresearch.ca one — no retry logic needed. Once both land, resume the
originally planned cross-check: match the 2,934 flagged
`SAT-01-12-2018_0617`/`_0619` window bucket IDs against `Syn.csv`'s
per-flow `Label` column (see §3 of `HANDOVER_2026-08-05.md`) to confirm or
rule out the val→test-gap label-quality theory before it goes in the
paper. The old `dataset/CIC_official_labels/fetch_csv_01_12.sh` /
`cic_cookies.txt` (direct cicresearch.ca path) are now superseded for
`CSV-01-12.zip` specifically but left in place — cookies may still be
useful if any other CICDDoS2019 CSV needs fetching later.

**2026-08-05 ~22:15 EEST — cross-check DONE, and it revises the earlier
theory rather than confirming it outright.** `CSV-01-12.zip` finished
transferring; extracted just `Syn.csv` (637MB, the SYN-flood attack-type
file, 1,582,681 flow rows). Ran inference with the continued checkpoint
(`model/flowpic_0p3_validated_v1_continued/resnext50_32x4d_best.ckpt`) on
just the two flagged stems' test images
(`scripts/diagnose_test_gap_infer.py` → `dataset/CIC_official_labels/flagged_windows_predictions.csv`)
and cross-referenced against `Syn.csv` (`scripts/diagnose_test_gap_crossref.py`).
Key facts established:

- Full test set with this checkpoint: 2,136 false positives (not the
  3,024 quoted in the prior session's note — that number was against a
  different/earlier checkpoint state). 2,071 of those 2,136 (97.0%) sit in
  the two flagged stems — **the 97% concentration finding replicates
  almost exactly**, so that part of the original diagnosis holds.
- Resolved the timezone question definitively: `window_features_real.csv`'s
  `t0_unix` is raw pcap epoch (UTC, unambiguous —
  `scripts/extract_window_features.py` reads packet timestamps directly).
  `Syn.csv`'s `Timestamp` column is `America/Halifax` local time (AST,
  UTC-4 on 2018-12-01 — DST ended 2018-11-04) — confirmed self-consistently:
  its own flow timestamps cluster in `13:30:30`–`13:34:27` local, matching
  `scripts/cic_ddos2019_syn_windows.json`'s documented `13:29`–`13:34`
  schedule almost exactly.
- **Important scope limit discovered**: `Syn.csv` only contains flows from
  that narrow ~4-minute window (`13:30:30`–`13:34:27` local =
  `17:30:30`–`17:34:27` UTC) — CICFlowMeter's per-attack-type export
  doesn't cover the full day, just the attack's active span (plus a
  handful of concurrent BENIGN flows). This means it can only confirm/deny
  labels for windows that fall inside that ~4-minute band.
- Of the 2,071 flagged false positives, only **192 fall within 60s of the
  official schedule window**; the other **1,879 range as early as 17:05:18
  UTC — up to 24 minutes before the documented attack even starts** — a
  time range `Syn.csv` has no data for at all, so can't be checked against
  it directly.
- Direct `Syn.csv` cross-reference (±2s tolerance): of the 2,071 flagged
  FPs, **86 have a matching official `Syn`-labeled flow** (genuine
  confirmed mislabels — these windows are provably real attack traffic
  wrongly marked "normal" by the schedule-boundary heuristic). **0 have a
  matching `BENIGN`-only flow. 1,985 have no official flow record at all**
  in that window (outside `Syn.csv`'s narrow coverage, so no verdict
  possible from this source).
- **Feature + visual check on the un-verifiable 1,985**: mean `syn_ratio`
  0.080 vs. 0.044 for genuine true-negatives in the same stem, mean
  `n_packets` 302.6 vs. 122.4 — measurably busier/SYNnier than normal, but
  a built contact sheet (8 sampled FP images vs. 4 confirmed-ddos vs. 4
  confirmed-normal, same stem) shows the FPs look like sparse background
  traffic — scattered faint dots — **not** the distinctive bright
  horizontal-line signature the confirmed-ddos tiles clearly show. So
  these are not visually attack-like; more likely unusually busy but
  genuinely normal traffic that the model over-generalizes on.

**Revised, more honest conclusion**: the prior session's framing (report
the boundary-excluded `f1_ddos=0.932` as "the real number," raw `0.375` as
pure label-boundary artifact) is **not well supported** — only 86/2,071
flagged FPs (4%) are provably mislabeled; the other 96% have no ground
truth to check either way, and what evidence exists (feature stats, visual
inspection) points to real normal traffic, not hidden attack traffic. The
defensible correction is much smaller: relabeling just the 86
confirmed-mislabeled windows (not excluding the whole two stems) moves
test `ddos_f1` from **0.419 → 0.453** and accuracy from **0.781 → 0.789**
(recomputed from the continued checkpoint's actual confusion matrix:
TP=822, FN=144, FP=2136, TN=7287, n=10389 — see
`scripts/diagnose_test_gap_crossref.py` output for the derivation). This
is a real, if modest, generalization gap on `01-12-2018` — genuinely
worth reporting as-is (raw `ddos_f1≈0.42`, corrected `≈0.45`) rather than
the much larger, less-supported `0.932` figure. Write this nuance into the
paper explicitly: "label quality partially explains the gap (4% of test
false positives), most of it is real."

**2026-08-05 ~22:05 EEST — launched the deprioritized bytedump-validated-v1
supervised run.** With the GPU free and the label cross-check done, picked
up the long-deprioritized item: `dataset/images_per_second_window_0p3_validated_v1`
(96,770 byte-dump PNGs, cluster-validated labels) had never been split.
`scripts/split_flowpic_blocked.py`'s stem-regex only matched flowpic
filenames (`..._b<id>.png`); bytedump filenames carry an extra
`_syn<N>.png` suffix (leftover from the old heuristic's naming). Widened
the regex in place to `_b\d+(?:_syn\d+)?\.png$` (backward compatible, still
matches flowpic filenames) rather than forking a duplicate script. `--dry-run`
reproduced the *exact* same per-class counts as the flowpic split
(train ddos=1402/normal=75376, val 165/9438, test 966/9423) — confirms the
stem-level assignment is identical across both encodings, so eval numbers
between them will be genuinely comparable, not confounded by different
held-out data. Applied for real (19,992 files moved). Launched
`scripts/train_resnext_per_second_0p3_validated_v1.sh` (unchanged from its
prior never-run form — ImageNet norm stats, focal loss, 40 epochs, matches
the existing primary checkpoint's recipe with only the label source
changed) in the background,
`model/per_second_0p3_validated_v1/train_2026-08-05_2200.log`. This is
what makes the long-planned 3-way comparison (primary vs.
validated-bytedump vs. validated-flowpic) finally possible once it
finishes — likely multi-hour given 224x224 images vs. flowpic's 64x64, not
expected to complete same-session.

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

**2026-08-04, FOURTH power loss, ~19:22 EEST (new session resuming work).**
Machine found freshly rebooted (`uptime` ~4min at session start, ~19:26).
The stage-1 supervised retrain had gotten to epoch 23/40 complete, died
mid-epoch-24 (step 1763/2399). Best checkpoint: **epoch 22**,
`f1_ddos=0.830380`, `acc=0.993023` (`model/flowpic_0p3_validated_v1/resnext50_32x4d_best.ckpt`,
saved 16:54) — better than any result logged from the prior 3 attempts today.

Given 4 outages in one day and `train_resnext.py` still has no mid-run
resume (see earlier entries), user decided **not** to relaunch the
40-epoch run a 4th time from scratch. Kept the epoch-22 checkpoint as
final for the supervised flowpic model and moved on.

While validating that checkpoint, found and fixed a real bug (not an
outage artifact): `scripts/eval_test_detailed.py` had no
`--norm-mean`/`--norm-std` flags, so it silently normalized with ImageNet
stats regardless of what the checkpoint was trained with. For this flowpic
checkpoint (trained with dataset stats ~1000x smaller than ImageNet's,
`0.0071 0.0003 0.0004` / `0.0592 0.0157 0.0145`), that made every eval
collapse to all-"normal" predictions (0 ddos recall) — looked exactly like
a corrupted/dead checkpoint but wasn't. Added the two flags, threaded them
into the `tr.make_dataset(...)` call. With correct stats:
- **val**: acc 0.993023, f1_ddos 0.830380 — matches the training-time
  validation log exactly, checkpoint confirmed healthy.
- **test** (held-out block, different sessions than train/val): acc
  0.704591, f1_ddos only 0.375076, ddos precision 0.233 (3024 false
  positives out of 9423 true-normal). Much worse than val — a real
  train/val→test generalization gap, not investigated further this
  session, worth a look before calling the flowpic model done.

Then launched the autoencoder stage (train→eval chain), which hadn't run
even once yet today across all 4 outages — no conflicting process/output
found first. `nohup ... & disown`, same caveat as always (survives
terminal death, not another power-off).

Note for next time: the script writes its own `training.log` inside
`--output-dir`, unbuffered, separate from whatever the shell redirects
`nohup`'s stdout to. The stdout redirect target sits empty for long
stretches (Python fully block-buffers stdout when it isn't a TTY) —
watch `training.log`, not the nohup redirect file, or it looks hung when
it isn't. `TO_RUN.sh` corrected to point at the right file.

**Chain completed 19:54 EEST, ran clean, no crash.** Results:
- Training: early-stopped at epoch 22/50, best checkpoint epoch 14
  (`val_mse=0.001920`). ~15min total (804s slow first epoch for dataset
  warmup/decode, then ~9.4s/epoch after).
- Eval (unsupervised anomaly detection via reconstruction error) —
  **weak, flag before treating as a finished baseline**:
  - val: `acc=0.291`, `roc_auc=0.288`, `f1_ddos=0.046` — accuracy far
    below the ~98.3% majority baseline, and **ROC AUC below 0.5** is the
    concerning part: reconstruction error is *anti-correlated* with the
    ddos label on val (ddos flows reconstruct better than normal ones,
    backwards from the premise of this baseline).
  - test: `roc_auc=0.601`, `f1_ddos=0.316` — better than val but still
    weak, and the val→test inconsistency itself suggests the threshold
    picked on val (`0.000879`, best-F1) doesn't transfer.
  - Not investigated further this session — could be a genuine negative
    result for this approach on this dataset, or a bug in
    `scripts/eval_flowpic_autoencoder.py`'s scoring/threshold direction
    (sub-0.5 AUC specifically shouldn't happen if the scoring direction
    is correct). Full report: `model/flowpic_autoencoder_v1/eval_report.json`.

**2026-08-04, ~20:00 EEST, two new workstreams per user direction** (deadline
2026-08-05 has flexibility, still worth using remaining time):

1. **Supervised: continue training past epoch 22.** New
   `scripts/train_resnext_flowpic_v1_continue.sh` — fine-tunes from
   `resnext50_32x4d_best.ckpt`'s weights (`--ckpt ... --strict-load`, NOT a
   true resume, fresh optimizer/schedule) with a lower peak LR (1.5e-4 vs.
   the original 5e-4) over a shorter 18-epoch cosine schedule, on the theory
   that the original run's LR was contributing to the val f1_ddos
   volatility seen late in that run (epoch 20 crashed to 0.059 before
   epoch 22 recovered to 0.830). Writes to
   `model/flowpic_0p3_validated_v1_continued/` (new dir, doesn't touch the
   epoch-22 checkpoint). Launched 20:13 EEST, confirmed weights loaded
   correctly (step-1 loss ~0.002, matching where the original run left off,
   not a from-scratch loss). Background via `nohup & disown`, same
   power-loss caveat as always.
2. **Unsupervised: Deep SVDD as a second baseline** (Ruff et al. 2018),
   since the autoencoder's reconstruction-MSE premise doesn't hold on this
   data (see above — ddos windows are sparser, not busier, than normal).
   Deep SVDD scores by distance to a learned center in embedding space
   instead, a different notion of "anomaly" than pixel reconstruction.
   New files: `scripts/deep_svdd_model.py` (bias-free encoder/decoder —
   no conv/dense bias, no BatchNorm affine — required so the network can't
   trivially collapse to a constant embedding; center eps-clamp against a
   zero coordinate; both are the standard collapse guards from the paper),
   `scripts/train_deep_svdd.py` (phase 1: pretrain encoder+decoder as an
   autoencoder on normal-only train, val-early-stopped on recon MSE; phase
   2: fix center from the pretrained encoder, fine-tune the encoder alone
   to minimize mean squared distance to it, val-early-stopped on that
   distance; logs `val_embed_std` every epoch as a collapse diagnostic —
   not used as the stopping metric, just a printed warning if it craters),
   `scripts/eval_deep_svdd.py` (imports `best_f1_threshold`/
   `report_at_threshold` from `eval_flowpic_autoencoder.py` rather than
   duplicating them — same val-sweep/test-report shape as the autoencoder's
   eval, for direct comparability), `scripts/train_deep_svdd.sh` (env
   wrapper, same CUDA_HOME/LD_LIBRARY_PATH pattern as the other `.sh`
   scripts — the autoencoder job hit a real bug earlier today from being
   invoked without this env, wrapper exists specifically to not repeat it).
   Smoke-tested end-to-end on CPU with a 40/12-image symlinked mini-dataset
   (2 pretrain + 2 SVDD epochs) — full pipeline runs clean: pretrain →
   center computation → fine-tune → checkpoint save/load → threshold
   search → JSON report, no exceptions, no collapse warning. Not yet
   launched for real on GPU — waiting for the supervised continuation run
   above to finish first (one job at a time on the single RTX 3070, per
   existing convention — avoids repeating the contention/instability risk
   flagged earlier).

**2026-08-04, ~22:15 EEST — continuation run finished + val→test gap
root-caused.**

Continuation run (item 1 above) completed all 18 epochs, no early stop.
Best checkpoint: **epoch 16**, val `acc=0.9948 f1_ddos=0.8656` (better than
the epoch-22 checkpoint's `0.830`). `train_resnext.py`'s own built-in
test-eval on that checkpoint: `acc=0.7805 f1_ddos=0.4190` (also better than
epoch-22's `0.375`, verified independently of `eval_test_detailed.py` since
this is the training script's own eval routine, same correct norm stats
throughout). **Both improved, but the val→test gap didn't close** —
`0.866→0.419` is essentially the same ~0.45 absolute gap as before. More
training bought incremental gains, not a fix.

Then ran a diagnostic (`/tmp/.../scratchpad/diagnose_test_gap.py` — scratch
only, not in repo) on the original epoch-22 checkpoint's test predictions,
joining per-image predictions back to source pcap stem via filename. **Root
cause found, and it's better news than the raw number suggested**: of the
3,024 false positives (normal→predicted ddos), **2,934 (97%) sit in exactly
two sub-pcap stems — `SAT-01-12-2018_0617` and `_0619`** — the same two
stems that contain the labeled-ddos windows themselves (FP rates 71.8% and
98.9% respectively). Every other stem in the entire held-out test day (450
stems, 5,370 normal windows spread across the whole day, zero overlap with
the attack) has only a **1.68%** false-positive rate. All of test's normal
images come from `01-12-2018` itself (not other days) — confirms it's a
same-day, attack-boundary-local effect, not a general cross-day
generalization failure.

This matches `reports/cluster_validation_2026-08-01.md`: `01-12-2018` (test's
ddos source) uses `resolution_mode=schedule_priority_low_syn_purity` —
schedule-based fallback labels (GMM ARI *negative*) because the
cluster-validation gate failed on this day — unlike `03-11-2018` (train/val's
ddos source), which cluster-validates cleanly (silhouette 0.64). The
schedule boundary is almost certainly narrower than the real attack traffic
(consistent with the earlier-noted "true SYN window is only ~5 minutes"),
so thousands of genuinely attack-adjacent windows in that busy period landed
on the "normal" side of an imprecise cutoff. The model is very likely
flagging real anomalies that are labeled wrong, not hallucinating.

Recomputed metrics excluding just those two boundary stems (same
`tp`/`fn`, only `same-day-other` normal windows as the negative class):
**`acc=0.979, precision_ddos=0.911, recall_ddos=0.953, f1_ddos=0.932`** —
better than the val numbers. **Not yet verified by eye** — this is strong
circumstantial evidence (concentration exactly in the attack-adjacent
stems, nowhere else in 450 other stems across the whole day), not a
confirmed relabel. Before this goes in the paper: spot-check a sample of
the 2,934 flagged images against confirmed-ddos ones from the same stem.
Recommend reporting **both** the raw test number and this boundary-excluded
number in the writeup, with the explanation — more defensible and more
interesting than picking one.

**2026-08-04, ~23:24 EEST — fetching official CICDDoS2019 labels to confirm
the mislabeling hypothesis properly.** User is downloading the official
per-flow CICFlowMeter CSV for `01-12-2018` (`CSVs/CSV-01-12.zip` from
cicresearch.ca) — this has the dataset's own authoritative `Label` column,
independent of our derived `window_features_real.csv` and independent of
the model's own predictions, so it's a real confirm/deny on the 2,934
flagged windows rather than more circumstantial evidence.

Site gates the download behind a session cookie (expires ~24h); user
exported one via Cookie-Editor. Endpoint confirmed to **not** support HTTP
Range requests (`curl -C -` → error 33 `HTTP server doesn't seem to
support byte ranges`) and doesn't report `Content-Length` (chunked) — so
no true resume and no upfront size estimate. Measured ~25-53KB/s, could
take hours depending on actual file size (unknown).

Built `dataset/CIC_official_labels/fetch_csv_01_12.sh` — retry loop that
fully restarts the download on any failure (up to 30 attempts, 30s
backoff) since resume isn't available, verifies zip integrity
(`unzip -tq`) before declaring success. Launched via `nohup & disown`
23:24 EEST, confirmed downloading (growing steadily). Cookie file at
`dataset/CIC_official_labels/cic_cookies.txt` — both it and the zip are
covered by the existing blanket `dataset/` gitignore rule, no secret-leak
risk. **Once this completes**: cross-reference the 2,934 flagged window
timestamps (`SAT-01-12-2018_0617`/`_0619`, bucket IDs available from the
image filenames / `window_features_real.csv`) against this CSV's official
per-flow labels to confirm or rule out the mislabeling hypothesis before
it goes in the paper.

**Also tried running the same fetch on the private staging server**
(`root@91.99.170.219`) in case it had better bandwidth — **doesn't work**,
different failure mode than the slow-but-working local download: TLS
handshake and request succeed, but the server returns **zero bytes** and
hangs indefinitely on this specific endpoint from that IP, despite general
`cicresearch.ca` access from there working fine (fast homepage load).
Almost certainly IP-based throttling/blocking of datacenter IPs on the
download endpoint specifically, not a network issue. Killed the stuck job
there (`pkill -f fetch_csv_01_12.sh`/`curl.*CSV-01-12`), confirmed nothing
left running remotely. **Local download is the only working path for
this file** — as of 23:34 EEST, 23.5MB in, still on attempt 1, ~39KB/s
average, no failures.

**Deep SVDD (item 2 above) launched for real, 23:34 EEST** —
`scripts/train_deep_svdd.sh`, GPU confirmed free (supervised continuation
finished, nothing else running). Chain log:
`model/flowpic_deep_svdd_v1_chain_2026-08-04_2334.log`. Phase 1
(autoencoder pretrain) started cleanly, dataset loaded correctly
(75,376 train / 9,438 val normal images, matches the other flowpic runs).

**Deep SVDD's first real run collapsed — hypersphere collapse, a training
bug not a data finding.** Finished clean (~7min total) but `eval_report.json`
showed `val roc_auc=0.00197`, predicting every single normal image as ddos
(`tn=0, fp=9438`) — the classic Deep SVDD failure mode: the encoder mapped
nearly all images (normal and ddos alike) to essentially the same point
near the center, so distance-to-center carries no signal. `val_embed_std`
(the collapse diagnostic) dropped monotonically the whole run, 0.380 →
0.0011 (>300x), never recovering. Root design bug: early-stopping was keyed
on `val_dist2` alone, but collapse *is* the global minimum of that exact
objective — it can't detect its own cause, which is why it ran the full 30
epochs looking like continuous improvement right up to total collapse.
**Unrelated to the label-quality issue above** — SVDD training (both
phases) only ever touches `normal/`-labeled images, no ddos label (correct
or mislabeled) is used anywhere in training, only in `eval_deep_svdd.py`'s
threshold sweep. Fixing the labels would not have prevented this.

Also worth noting: extreme input sparsity (the same characteristic behind
the autoencoder's sparsity-inversion finding) plausibly makes this
particular collapse mode easier to fall into than on denser benchmark
data (CIFAR/MNIST) Deep SVDD is usually demonstrated on — a bias-free
network fed mostly-zero input naturally drifts toward near-zero output
unless weights are specifically tuned to amplify the sparse signal.

Fixed in `scripts/train_deep_svdd.py`: (1) new `--min-embed-std` floor
(default 0.05) — a checkpoint is only accepted as "best" if `val_embed_std`
stays at/above it, and training now hard-stops the first epoch it drops
below (collapse doesn't recover, no point continuing or using
no-improve-patience for it); (2) modest weight-decay bump (`1e-6 → 1e-4`,
deliberately conservative, not the 10-100x initially considered — heavier
weight decay pulls weights toward zero, which is the same direction as
collapse here given the sparse-input dynamic above, so overcorrecting could
make it worse, not better). The `--min-embed-std` mechanism is the primary
fix; weight decay is secondary/experimental. Old collapsed run preserved at
`model/flowpic_deep_svdd_v1_collapsed_2026-08-04/` for the record. Relaunched
23:54 EEST, chain log `model/flowpic_deep_svdd_v1_chain_2026-08-04_2350.log`.

**Fix confirmed working, epoch 6/`std=0.051` selected instead of collapsing
further** — but the actual result is weak: `val roc_auc=0.287, f1_ddos=0.044`
/ `test roc_auc=0.519, f1_ddos=0.011`. `val roc_auc=0.287` lands suspiciously
close to the plain autoencoder's own `0.288` — likely the same underlying
cause: the SVDD encoder starts from the phase-1 AE-pretrained weights, and
with training cut short at epoch 6 to avoid collapse, it hasn't moved far
from that initial sparsity-correlated representation (ddos windows are
sparser than normal, see the autoencoder finding above).

**2026-08-05, ~00:05 EEST — tried a 5x higher SVDD-phase LR (1e-4→5e-4)** to
see if faster movement per epoch could pull the encoder away from the
AE-inherited representation before hitting the collapse floor. **Made it
worse, not better** — collapsed even faster (epoch 4 vs. epoch 7) and both
metrics dropped: `val roc_auc=0.242` (vs. `0.287`), `test roc_auc=0.467`
(vs. `0.519`). This is actually informative: it confirms the problem isn't
"not enough movement per step," it's that there's a fundamentally short
pre-collapse window regardless of step size, not enough of it to escape the
inherited representation. Reverted to the lr=1e-4 result as current best
(`model/flowpic_deep_svdd_v1/`); the lr=5e-4 attempt preserved at
`model/flowpic_deep_svdd_v1_worse_lr5e-4_2026-08-05/` for the record.

**Conclusion for now**: Deep SVDD's collapse bug is genuinely fixed
(methodological success, the `--min-embed-std` guard works as designed),
but on this dataset it doesn't currently outperform the plain autoencoder
— both unsupervised approaches plausibly fail for the same underlying
reason (sparsity-correlated representations, and ddos windows being
sparser than normal inverts the usual anomaly-detection assumption
regardless of which specific method is used). This itself is a more
interesting, more defensible finding for the writeup than either result
alone. Not tried: training the SVDD encoder from random init instead of
AE-pretrained (a more structurally different lever than LR, would avoid
inheriting the AE's representation entirely, but loses the paper's
"standard Deep SVDD init procedure" framing) — worth a future attempt if
there's time, not pursued further this session.

**2026-08-05, ~23:15 EEST — FIFTH power loss today (new session resuming
work).** Machine found freshly rebooted (`uptime` ~3min at session start).
The bytedump-validated-v1 training (§2 of `HANDOVER_2026-08-05.md`, launched
22:05 EEST) had gotten through epochs 1-8 (all steps of epoch 8 completed,
died mid-checkpoint-write — `resnext50_32x4d-8_2399.tmp` left as a 0-byte
file) but never improved past epoch 2's `f1_ddos=0.0807` (saved 22:27) —
epochs 3-8 oscillated between ~0.03 and ~0.08, the same known volatility
pattern this recipe showed on the flowpic run before it eventually recovered
to `f1_ddos=0.87` by epoch 16. The `CSV-03-11.zip` pull also died (365MB of
919MB, `.part` file, no resume support — full progress lost, not
re-launched, nothing is blocked on it per the existing handover note).

Before relaunching, did a correctness pass on the training code per user
request (make sure the code is right before spending more GPU time on it) —
`notebook/train_resnext.py` is untouched by git status (not modified since
the flowpic run that already validated this exact recipe), so no regression
was introduced by the outage itself. Verified directly: class-weight math
(`[1.9954, 0.00464]` for counts `[1402, 75376]` with `minority_boost=8`)
recomputes exactly to the value the log printed; sample images from
train/ddos, train/normal, and test/ddos all decode as valid non-corrupt
grayscale 224x224 PNGs with real varied pixel content (not blank/corrupt);
`scripts/eval_test_detailed.py` still has the `--norm-mean`/`--norm-std`
flags from the earlier bug fix (relevant for the eventual 3-way comparison,
though this particular run uses default ImageNet stats intentionally, to
match the existing primary checkpoint's recipe for a valid comparison — not
a bug). No code issues found — the epoch 1-8 oscillation is expected
recipe behavior, not a defect.

Given no real improvement occurred past epoch 2 of 8, and `train_resnext.py`
has no optimizer/LR-schedule checkpointing anyway (resuming from epoch-2
weights would not meaningfully differ from a fresh cosine schedule),
restarted from scratch rather than resuming. Moved the crashed run's
checkpoints/log aside to
`model/per_second_0p3_validated_v1_crashed_2026-08-05_2308/` (not deleted).
Relaunched via `nohup bash scripts/train_resnext_per_second_0p3_validated_v1.sh
& disown`, confirmed alive past context-init and past epoch 1 step progress
(PID 5083). Log: `model/per_second_0p3_validated_v1/train_2026-08-05_2325.log`.
Same caveat as always: survives terminal death, not another power-off.
`TO_RUN.sh` updated with current log path and a generalized 6th-outage
recovery recipe.

**2026-08-05 ~23:35 EEST onward — user flagged a scheduled 1:55am power
outage; scope cut to what's actually finishable, then executed same
session.** User said the machine's power would be cut at 1:55am (not
another surprise outage — planned), which meant the bytedump-validated-v1
run (started 23:19, only at epoch 3-4 with cold disk cache making it slower
than the pre-outage attempt) had no realistic chance of reaching the
~epoch-16+ point where this recipe historically stabilizes. Also tested
`git push` proactively (via a disposable `test/push-access-check` branch,
cleaned up after) and found it **did not work** — no credential helper
configured, `fatal: could not read Username for 'https://github.com'`.
Fixed by having the user run `gh auth login` (after routing around a
pre-existing broken `cudnn-local-repo` apt signature by using
`sudo snap install gh --classic` instead of apt); re-tested push
end-to-end afterward (branch pushed, verified on GitHub, deleted both
locally and remotely) — confirmed working.

Given the time crunch, made an explicit scope decision with the user rather
than silently deciding it: **do not put the bytedump-validated-v1 run's
number in the paper if it's cut short** — an undertrained checkpoint
doesn't tell you the byte-dump encoding is worse, only that it had fewer
epochs, so including it would misrepresent the ablation. Killed that
training process outright (PID 5083) once this was decided — no reason to
keep burning GPU/disk on a result that wouldn't be used, and the scheduled
outage would kill it anyway. The 3-way encoding comparison (primary vs.
validated-bytedump vs. validated-flowpic) is **not completed** and is not
in the paper; only two of the three legs exist (primary, done long ago;
validated-flowpic, completed 2026-08-04). This is intentionally left as
future work, not silently dropped — see `reports/label_validation_and_generalization_gap_2026-08-06.md`'s
closing section.

**Recovered the actual Overleaf `.tex` sources for the two loose PDFs at
repo root** — the 2026-07-26 session's open question ("recover the Overleaf
sources") turned out to already be answerable: the user had
`DDoS_Attack_Detection_Using_ResNeXt50_32x4d_with_MindSpore.zip` and
`..._paper.zip` sitting at repo root the whole time (same names as the
PDFs, just not `.tex` — never previously unzipped/inspected). Extracted
both:
- `..._with_MindSpore.zip` → nearly identical to `research/main.tex` (only
  abstract wording differs, 928 vs 925 lines) — this is genuinely the same
  report, not a separate document. No action needed beyond noting it.
- `..._paper.zip` → the **real** SYNShield IEEE conference paper
  (`paper.tex`, real author block: Nehme, Al Ghoush, Farhat with the known
  placeholder `XXXXXXX@bau.edu.lb` email) — this is a genuinely different
  document from `ict_innovation_overleaf/main.tex`, which turned out to be
  an unrelated, still-templated (`\TeamName` placeholder) submission for a
  different competition track. Brought the real paper into the repo,
  tracked, at `research/synshield_paper/` (`paper.tex` +
  `IEEEtran.cls`/`.bst` + `references.bib` + figures) — the first time
  this source has been under version control.

**Paper edits made this session** (both `research/main.tex` and
`research/synshield_paper/paper.tex`), using only already-complete results
(the flowpic-validated model's numbers, finalized 2026-08-04/05 — nothing
from tonight's cut-short run):
- Fixed the long-flagged `SYNSheild` → `SYNShield` typo (3 occurrences,
  `research/synshield_paper/paper.tex` only — the misspelling wasn't present
  in `research/main.tex`).
- Added a new subsection to both documents (`research/main.tex` §"Label-quality
  validation and cross-day generalization", `paper.tex`
  §"Label-Quality Validation and Cross-Day Generalization") presenting the
  cluster-validated-labels + FlowPic + cross-day-holdout methodology, the
  raw-vs-corrected `f1_ddos` numbers (0.419 → 0.453), and — `research/main.tex`
  only, omitted from the paper for space — the two unsupervised baselines as
  a reported negative result.
- Updated `research/main.tex`'s Contributions, Limitations, and Conclusion
  sections; updated `paper.tex`'s Limitations bullet accordingly.
- Added `shapira2019flowpic` (FlowPic) to both `references.bib` files and
  `ruff2018deep` (Deep SVDD) to `research/references.bib`.
- **Not verified by compiling** — no `pdflatex`/`texlive` installed on this
  machine, confirmed before starting. Did a manual brace/environment-balance
  sanity check (Python script counting `\begin`/`\end` pairs and curly-brace
  depth, comments stripped) on both files pre- and post-edit; the count is
  unchanged by the edits (a pre-existing depth-2 imbalance in
  `research/main.tex` predates this session, confirmed against
  `git show HEAD:research/main.tex`). This is **not** a substitute for an
  actual compile — recommend running it through Overleaf or a local
  `pdflatex` pass before treating it as submission-ready.
- `research/synshield_paper/` does not include a recompiled `paper.pdf` —
  the zip's original `paper.pdf` predates tonight's edits and was
  deliberately not copied in stale/unlabeled.

## Next steps (in order)

1. ~~Sequential transfer of all 5 PCAP zips finishes~~ — **done 2026-08-04
   05:29 EEST**, see above. All 5 zips present + verified, both image sets
   complete (96,770 PNGs each, matching label ground truth exactly).
2. ~~Run `scripts/generate_validated_images_batch.sh <zip>` per zip~~ — done
   for all 5 as part of the above.
3. ~~`scripts/split_train_val_test.py` on both new dataset roots~~ — done
   (blocked split, see 2026-08-04 commit history).
4. ~~`scripts/compute_flowpic_norm_stats.py` on the flowpic train split~~ —
   done, plugged into `scripts/train_resnext_flowpic_v1.sh`.
5. ~~Retrain supervised flowpic model~~ — **stopped at epoch 22/40** after
   the 4th power loss (see above); accepted as final rather than risk a 5th
   from-scratch restart. `scripts/train_resnext_per_second_0p3_validated_v1.sh`
   (bytedump variant) status not reconfirmed this session — check before
   assuming it's still current.
6. ~~Autoencoder train+eval chain~~ — **done 2026-08-04 19:54 EEST**, ran
   clean. Results are weak (see above) — do NOT report as a working
   unsupervised baseline without further digging.
7. ~~Investigate the val→test generalization gap~~ — **root cause found
   2026-08-04 ~22:15 EEST, then REVISED 2026-08-05 ~22:15 EEST after the
   official-label cross-check landed.** 97% concentration in the two
   flagged stems still holds, but it is only **partially** a
   schedule-label-boundary artifact: only 86/2,071 flagged false positives
   (4%) are provably mislabeled against the official `Syn.csv`; the rest
   have no official ground truth available (outside `Syn.csv`'s narrow
   ~4min coverage) and look, both by feature stats and visual spot-check,
   like real (if busier-than-typical) normal traffic — not hidden attack
   traffic. **Defensible corrected metric: `ddos_f1≈0.453`** (relabeling
   only the 86 confirmed mislabels), not the earlier `0.932`. See the
   2026-08-05 ~22:15 EEST entry above for full derivation. This is now a
   genuine, if partially explained, generalization gap worth reporting
   honestly in the paper.
8. ~~Investigate the autoencoder's sub-0.5 val ROC AUC (0.288)~~ — **root
   cause found, not a code bug.** `scripts/eval_flowpic_autoencoder.py`'s
   scoring/threshold logic is correct (labels, score direction,
   `pred = scores >= t` all consistent). Sampled 300 val images per class
   directly: ddos-labeled 0.3s FlowPic windows are ~3.7x *sparser* than
   normal ones (nonzero-pixel frac 0.0159 vs 0.0589, mean pixel 0.0136 vs
   0.0187) — a SYN flood in a tight 0.3s window produces a uniform,
   concentrated pattern; normal traffic in the same window is busier
   (more varied packet sizes/timing). The autoencoder converged almost
   immediately to a near-trivial low-error solution (`train_mse` dropped
   ~20x from epoch 1→2, then plateaued) — consistent with learning to
   reconstruct sparse/background patterns well. Since ddos images are
   sparser, not busier, than normal, they're *easier* to reconstruct,
   inverting the "anomaly = high reconstruction error" assumption this
   whole baseline relies on. **This is a genuine negative result for
   reconstruction-based unsupervised detection on the FlowPic 0.3s
   representation** — worth stating as such in the writeup, not silently
   dropped. Not investigated further: whether a different
   architecture/bottleneck size, or scoring by *inverse* error, or a
   different window size, would fix it.
9. ~~Supervised continuation run past epoch 22~~ — **done ~22:15 EEST**,
   best checkpoint epoch 16 (val `f1_ddos=0.8656`), see above. Both val and
   test improved slightly over the epoch-22 checkpoint but the val→test gap
   itself didn't close (root-caused separately, see item 7).
10. ~~Deep SVDD real GPU training+eval~~ — **done, multiple passes**: first
    run collapsed (bug, fixed), second run (lr=1e-4) is current best
    (`model/flowpic_deep_svdd_v1/`, val `roc_auc=0.287`), third run
    (lr=5e-4, testing whether faster movement escapes the AE-inherited
    representation) made it worse and was reverted. See above for full
    detail. Current conclusion: doesn't beat the autoencoder, likely same
    underlying cause. Untried lever if revisited: SVDD encoder from random
    init instead of AE-pretrained.
11. ~~Spot-check the flagged `SAT-01-12-2018_0617`/`_0619` windows against
    the official CICDDoS2019 per-flow labels~~ — **done 2026-08-05
    ~22:15 EEST**, see item 7 above for the (revised) conclusion.
12. **3-way comparison (primary vs. validated-bytedump vs. validated-flowpic)
    — NOT completed, intentionally dropped for this cycle.** The
    validated-bytedump leg was killed 2026-08-05 ~23:55 EEST at epoch ~4,
    undertrained, per an explicit user + Claude scope decision ahead of the
    1:55am scheduled outage: don't put a cut-short checkpoint's number in
    the paper as if it were a real ablation result. If resumed later: relaunch
    `scripts/train_resnext_per_second_0p3_validated_v1.sh` fresh (crashed
    attempt preserved at
    `model/per_second_0p3_validated_v1_crashed_2026-08-05_2308/` and
    `model/per_second_0p3_validated_v1/train_2026-08-05_2325.log`, both
    superseded, kept for reference only), let it run to natural
    early-stopping (no time pressure next time), then eval all three
    checkpoints and write the comparison. The dataset split is already done
    (`dataset/images_per_second_window_0p3_validated_v1/{train,val,test}`) —
    only the training + eval + write-up remain.
13. ~~Paper/report text rewrite for the label-validation finding, SYNShield
    spelling, recovering the Overleaf `.tex` sources~~ — **done 2026-08-06**,
    see the entry above and `reports/label_validation_and_generalization_gap_2026-08-06.md`.
    Still open, genuinely deferred (no action taken, no new information this
    session): **Mohammed Farhat's placeholder email**
    (`XXXXXXX@bau.edu.lb` in `research/synshield_paper/paper.tex`) — needs
    the real address from Farhat or the team, not something to guess/invent.
    `install.sh` disposal also still untouched (still sitting at repo root,
    still believed to be Anthropic's CLI installer dropped in by accident,
    not part of this project) — low priority, unrelated to the paper.

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
