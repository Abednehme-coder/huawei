# Handover: Unsupervised FlowPic Autoencoder — A-Z

Self-contained handover for a **different agent/session** to own this one
piece of work end-to-end: train, calibrate, evaluate, and report the
unsupervised (no-ddos-labels) baseline that the paper compares against the
supervised ResNeXt-50 classifier. You don't need any other context from this
repo's other handover docs to execute this — everything needed is below —
but `WORK_LOG.md` and `HANDOVER_2026-08-04.md` have the wider project story
if you want it.

## 1. What this is and why it exists

The paper's current main new result is a **supervised-vs-unsupervised
comparison** on the same held-out test set, same image encoding (FlowPic:
each 0.3s traffic window → a 64×64×3 packet-size/arrival-time histogram
image). The supervised side (ResNeXt-50, full ddos labels) is a separate,
already-running piece of work — not your concern. Your job is the
**unsupervised side**: a small convolutional autoencoder trained ONLY on
normal-labeled traffic windows, which never sees a single ddos-labeled image
during training. At eval time, reconstruction error becomes the anomaly
score (a window that looks like normal traffic reconstructs well; an attack
window, being out-of-distribution for the model, reconstructs poorly). A
threshold on that error, calibrated on val (still without training on
labels — just picking a decision boundary), turns it into a classifier
comparable to the supervised model.

**Deliverable**: `model/flowpic_autoencoder_v1/eval_report.json` (val +
test metrics: accuracy, precision/recall/F1 on the ddos class, ROC-AUC,
PR-AUC, confusion matrix) plus a short written summary of those numbers,
ideally with a 1-2 sentence comparison against the supervised model's test
numbers (ask in this repo's WORK_LOG.md / the other agent's session for
those once both are done — don't block on it, report yours independently
first).

## 2. Everything you need, all already built and tested

No new code should be needed — this is a "run it and interpret the output"
task, not a "build it" task. If something is broken, fix the bug, don't
redesign the pipeline.

- **Model**: `scripts/flowpic_autoencoder_model.py` — `ConvAutoencoder`
  class (64×64×3 → 4×4×256 bottleneck → 64×64×3, sigmoid output, ~4 conv +
  4 transpose-conv layers, BatchNorm+ReLU throughout). Also has
  `make_image_only_dataset()`, the shared data-loading helper (loads one
  class subdir at a time, no labels attached — reused for both training and
  per-class eval scoring).
- **Train script**: `scripts/train_flowpic_autoencoder.py` — trains on
  `{data-root}/train/normal/*.png` only, validates (reconstruction MSE,
  still unsupervised) on `{data-root}/val/normal/*.png`, saves
  `{output-dir}/autoencoder_best.ckpt` whenever val MSE improves, early-stops
  after `--early-stop` (default 8) non-improving epochs, max `--epochs`
  (default 50).
- **Eval script**: `scripts/eval_flowpic_autoencoder.py` — loads a
  checkpoint, scores val (both classes, this is the ONLY place labels touch
  anything) to pick the best-F1 threshold, then reports val+test metrics at
  that threshold, writes `{output-dir}/eval_report.json`.
- **Data**: `dataset/images_flowpic_0p3_validated_v1/{train,val,test}/{ddos,normal}/*.png`
  — already generated, split, and verified. Counts: train 76,778 (1,402
  ddos + 75,376 normal), val 9,603 (165 ddos + 9,438 normal), test 10,389
  (966 ddos + 9,423 normal). Total size on disk: **386MB** (small — trivial
  to copy/transfer if running this on a different machine — see §3 for exact
  transfer commands). The split is blocked by source-pcap file (not random
  per-image) specifically so adjacent-window leakage across train/test
  can't happen — don't re-split, don't shuffle across the class/split
  boundaries, use it as-is.
- **Environment**: MindSpore (GPU build), CUDA 11.6, Python venv at
  `.venv/bin/python` in repo root. **Important gotcha that has bitten this
  exact pipeline before**: invoking the `.py` scripts directly without
  setting `CUDA_HOME`/`LD_LIBRARY_PATH` first causes MindSpore to silently
  fail to find GPU libs and error out on `device_target=GPU`. Always set:
  ```bash
  export CUDA_HOME=/usr/local/cuda-11.6
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  ```
  before running either script directly (there's no `.sh` wrapper for these
  two, unlike the supervised training script — you have to set this by hand).

## 3. Getting the data onto another device

**You do NOT need the raw PCAP zips for this task.** Those are huge
(~23GB total across 5 files, in `dataset/PCAPs_raw/` on this machine) and
are only the *source* material for image generation — a separate, already-
completed step. All this task needs is the 386MB of already-generated
FlowPic PNGs plus the labels baked into their directory structure
(`{split}/{ddos,normal}/*.png` — class is encoded in the path, no separate
label file to bring over).

From this machine (`abed@<this-host>`), package and transfer just that
directory:

```bash
# On this machine — package the split (386MB, ~a few seconds)
cd /home/abed/Documents/huawei/dataset
tar -czf images_flowpic_0p3_validated_v1.tar.gz images_flowpic_0p3_validated_v1

# Then transfer to the other device, e.g. via scp (adjust user@host/path):
scp images_flowpic_0p3_validated_v1.tar.gz <user>@<other-device>:/path/to/repo/dataset/

# On the other device — unpack into the same relative location
cd /path/to/repo/dataset
tar -xzf images_flowpic_0p3_validated_v1.tar.gz
```

If instead you're setting up on a cloud VM with no direct SSH path back to
this machine, upload the tarball to whatever storage the VM can pull from
(GCS bucket, a temporary transfer service, etc.) — no specific service is
set up for this yet, use whatever's easiest given where "another device" is.

**If for some reason the 386MB split isn't available and needs to be
regenerated from scratch** (shouldn't be necessary, but as a fallback): the
raw PCAP zips live in `dataset/PCAPs_raw/` on this machine, originally
pulled from a private staging server (`root@91.99.170.219:/root/dataset/PCAPs/`,
see `dataset/PCAPs_raw/parallel_fetch_generic.sh` for the fetch mechanism —
SSH-key-based, key at `~/.ssh/agent_key`) which mirrors the public
CICDDoS2019 dataset. Regenerating from raw zips means re-running the full
pipeline (`scripts/generate_validated_images_batch.sh` →
`scripts/split_flowpic_blocked.py`) — a multi-hour undertaking, not a
quick fallback. Flag back to the requester before going down this path
rather than assuming it's expected.

## 4. Exact commands

```bash
cd /home/abed/Documents/huawei
export CUDA_HOME=/usr/local/cuda-11.6
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Train (defaults point at the right data-root/output-dir already; only
# override if you deliberately want a different location, see §5 below)
.venv/bin/python scripts/train_flowpic_autoencoder.py \
  --data-root dataset/images_flowpic_0p3_validated_v1 \
  --output-dir model/flowpic_autoencoder_v1

# Eval (run only after training produces autoencoder_best.ckpt)
.venv/bin/python scripts/eval_flowpic_autoencoder.py \
  --data-root dataset/images_flowpic_0p3_validated_v1 \
  --ckpt model/flowpic_autoencoder_v1/autoencoder_best.ckpt \
  --output-dir model/flowpic_autoencoder_v1
```

Run both in the foreground and watch, or background with
`nohup ... > logfile 2>&1 & disown` if you need it to survive a session end.
Either is fine — this job is much shorter than the supervised one (see
timing below).

## 5. ⚠️ Read before running: this may already be running / about to run

There is **already a queued, automatic run of this exact same training+eval
pipeline** on this same machine, chained to start the moment a currently-running
supervised ResNeXt-50 job finishes (single RTX 3070, one job at a time,
that's why it's queued rather than parallel). As of this handover being
written, that supervised job is mid-training (~epoch 22/40) with no fixed
finish time (could be anywhere from ~30min to ~2h away depending on early
stopping). Check the state of the world before you start:

```bash
ps aux | grep "[t]rain_resnext.py\|[t]rain_flowpic_autoencoder"
ls -la /home/abed/Documents/huawei/model/flowpic_autoencoder_v1/
```

- If you see a `train_flowpic_autoencoder.py` process already running
  (i.e. the queued chain got there first) — **don't start a second one**,
  it'll write to the same output dir and corrupt/race the checkpoint file.
  Just watch/wait: `tail -f model/flowpic_autoencoder_v1/stdout.log`.
- If `train_resnext.py` is still running and you want to run the
  autoencoder **now**, in parallel, rather than wait: that's fine
  functionally (this model is tiny, ~64×64 images, nowhere near as
  GPU-hungry as the ResNeXt job), but you'll be sharing one GPU with the
  supervised job, so both slow down somewhat, and **you must use a different
  `--output-dir`** (e.g. `model/flowpic_autoencoder_v1_parallel/`) so you
  don't collide with the queued run's own eventual write to
  `model/flowpic_autoencoder_v1/`. Reconcile afterward — pick whichever
  finished run's `eval_report.json` you trust (they should converge to
  similar numbers; if they don't, that's itself worth flagging, could
  indicate GPU-contention instability).
- If this is being run on a **different machine entirely** (e.g. cloud, per
  earlier discussion in this repo's chat about running it on Google Cloud
  to parallelize/avoid local power-outage risk): no collision concern, just
  transfer `dataset/images_flowpic_0p3_validated_v1/` per §3 and run
  independently. Bring the results (`eval_report.json` +
  `autoencoder_best.ckpt`, the latter is small — check size, but conv
  autoencoders are typically a few MB, nothing like the 276MB ResNeXt
  checkpoints) back into this repo's `model/flowpic_autoencoder_v1/` when
  done, and update `model/flowpic_autoencoder_v1/` to be findable for
  whoever writes the final comparison report.

## 6. Known pitfalls (all previously hit today, on this exact pipeline)

- **No resume-from-checkpoint.** If training is interrupted (power loss,
  session death, whatever), there is no path to continue from a mid-run
  state — only `autoencoder_best.ckpt` (best epoch's weights) persists and
  is independently usable. A restart means retraining from epoch 1, not
  resuming. This has cost real time today (three separate mains power
  outages hit this machine).
- **Env vars are mandatory for direct invocation** — see §2, this exact
  mistake (missing `CUDA_HOME`) already caused a failed run today.
- **This machine has had three unplanned power outages today** (confirmed
  mains issue, not thermal/hardware — not something to investigate, just
  something to plan around if running here). If running elsewhere (cloud),
  this isn't a concern.

## 7. Expected timing

From a partial run earlier today (before an outage interrupted it): epoch 1
took ~296s (~5 min) on this machine's RTX 3070. Default is 50 epochs max,
early-stop patience 8. Realistic estimate: **~1.25-1.7 hours** if it
converges/early-stops in the 15-20 epoch range (typical for a small
autoencoder), up to **~4.1 hours** worst case if it runs the full 50. Eval
itself (§3, second command) is fast — a few minutes, it's just a forward
pass over val+test, no training.

## 8. Definition of done

1. `model/flowpic_autoencoder_v1/eval_report.json` exists and has both
   `val` and `test` keys with real (non-NaN, non-zero-everything) metrics.
2. Sanity-check the test result isn't degenerate before reporting it as a
   real result — e.g. if `recall_ddos` or `precision_ddos` in the test block
   is exactly 0 or the confusion matrix shows the model predicting one class
   for everything, that's a sign the threshold or training went wrong, not
   a real "unsupervised is bad" finding. Worth a sanity plot/spot-check of a
   few reconstruction errors if that happens, rather than reporting it at
   face value.
3. Report back (in whatever form the requester wants — chat summary,
   updated `WORK_LOG.md` entry, etc.) the test-set accuracy/F1/ddos-F1/ROC-AUC
   numbers, and where the checkpoint + report file live on disk.
