# Label-quality validation and cross-day generalization gap

Date: 2026-08-06 (session started 2026-08-05 evening). Supersedes the informal
conclusions in `WORK_LOG.md`'s 2026-08-04/05 entries and
`HANDOVER_2026-08-05.md` section 1 — this is the write-up of that work for
the paper. Read those for the full blow-by-blow; this is the condensed,
citable version.

## Why this exists

The primary model (`model/per_second_0p3_focal_v2/`, 98.1% test accuracy,
0.937 macro F1, reported in `research/main.tex` and the SYNShield conference
paper) is trained on labels from a raw SYN-count heuristic: PCAPs with more
than a fixed threshold of TCP SYN packets are labeled DDoS. A reasonable
reviewer objection is that this heuristic could be doing most of the work —
the model might just be re-deriving a threshold rule rather than learning a
genuinely useful visual representation. This work tests that objection
directly rather than leaving it unaddressed.

## What was built

1. **Unsupervised, cluster-validated labeling.** K-means (k=2) on per-window
   flow features (packet counts, SYN ratio, inter-arrival statistics), with
   a *flood-signature gate*: the picked "attack" cluster is only accepted if
   its mean SYN ratio exceeds the other cluster's. This gate exists because
   the original ungated version could pass its silhouette-score check while
   splitting on unrelated traffic volume rather than attack behavior — found
   on `01-12-2018`, whose true SYN-flood window is only ~5 minutes long, so
   whole-day k=2 clustering tends to separate on something else. Falls back
   to schedule-based labels when the gate fails.
2. **FlowPic-style encoding** (Shapira & Shavitt, 2019): each 0.3s window
   becomes a 64×64×3 packet-size × arrival-time histogram (all-packets /
   SYN-only / ACK|FIN|RST channels), log-scaled against a fixed global clip.
   Chosen specifically to avoid the natural-photo assumptions (ImageNet
   augmentation, ImageNet normalization stats) the byte-dump encoding
   carries despite being trained from scratch on packet bytes, not photos.
3. **Cross-day held-out split**, stricter than the primary model's
   same-domain split: the entire `01-12-2018` attack day (3 sub-pcap stems,
   966 DDoS windows) is test, held out whole. Train is the other attack day
   (`03-11-2018`, 8 stems, 1402 DDoS windows) plus normal-only captures.
   Split by whole source PCAP, never by individual window, to avoid
   adjacent-window leakage across splits.

## Result

Model: `model/flowpic_0p3_validated_v1_continued/resnext50_32x4d_best.ckpt`
(epoch 16 of an 18-epoch continuation run; best on validation DDoS F1).

| Split | Metric | Value |
|---|---|---|
| Validation | accuracy | 0.9948 |
| Validation | f1_ddos | 0.8656 |
| Test (cross-day, raw) | accuracy | 0.7805 |
| Test (cross-day, raw) | f1_ddos | 0.419 |

A real, substantial same-checkpoint gap between validation and held-out
cross-day test.

## Diagnosis

97% of the 2,136 test false positives concentrate in exactly the two
sub-pcap stems (`SAT-01-12-2018_0617`, `_0619`) that contain the day's
labeled DDoS windows — every other stem in the 450-stem held-out day has a
1.68% false-positive rate. This narrowed the question to whether those two
stems' labels are trustworthy.

Cross-referenced the 2,071 flagged windows in those two stems against
CICFlowMeter's **official** per-flow `Label` column for `01-12-2018`
(`Syn.csv`, from the dataset's own CSV export) — independent of both our
derived labels and the model's own predictions. Findings:

- `Syn.csv` only covers a narrow ~4-minute window (the attack's documented
  active span); it has no records at all for most of the flagged windows'
  timestamps, so it can only confirm or deny a subset.
- Of the 2,071 flagged windows, **86 have a matching official `Syn`-labeled
  flow** (±2s tolerance) — genuinely mislabeled by our schedule-boundary
  heuristic, real attack traffic marked "normal" because the documented
  attack schedule window is narrower than the actual attack. **0** match a
  `BENIGN`-only flow. **1,985** have no official record at all (outside
  `Syn.csv`'s narrow coverage).
- For the 1,985 unverifiable windows: feature stats show them measurably
  busier than confirmed true negatives in the same stem (mean `syn_ratio`
  0.080 vs. 0.044, mean `n_packets` 302.6 vs. 122.4) — but a visual
  contact-sheet comparison (8 sampled false positives vs. confirmed-DDoS vs.
  confirmed-normal, same stem) shows they resemble sparse background
  traffic, not the bright horizontal-line SYN-flood signature confirmed-DDoS
  images clearly show.

## Corrected metric

Relabeling only the 86 confirmed mislabels (not excluding the two stems
outright, which the evidence doesn't support):

| Metric | Raw | Corrected (86 relabeled) |
|---|---|---|
| Accuracy | 0.7805 | 0.7889 |
| DDoS precision | 0.278 | 0.307 |
| DDoS recall | 0.851 | 0.863 |
| **DDoS F1** | **0.419** | **0.453** |

## Conclusion for the paper

Report this as a real, only *partially* label-driven cross-day
generalization gap: 4% of test false positives are confirmed label-boundary
artifacts; the rest reflect genuine model imprecision on busier-than-typical
normal traffic, not a labeling problem. This is a more modest but far more
defensible claim than either (a) ignoring the gap, or (b) the initially
considered framing of excluding the two stems outright (which would have
implied a `f1_ddos≈0.93`, not well supported once the official labels were
checked).

## Unsupervised baselines (negative result, reported honestly)

Two anomaly-detection baselines were also attempted on the same
FlowPic-encoded, cluster-validated-label data, trained on normal-only
windows with no DDoS label ever seen — as a check on whether unsupervised
detection could sidestep the labeling question entirely:

- **Convolutional autoencoder** (reconstruction error as anomaly score):
  test ROC AUC 0.601, f1_ddos 0.316. Validation ROC AUC was actually *below*
  0.5 (0.288) — root-caused to DDoS windows in this 0.3s representation
  being ~3.7x *sparser* than normal windows (nonzero-pixel fraction 0.0159
  vs. 0.0589), the opposite of the usual "anomaly = busier/higher error"
  assumption reconstruction-based detection relies on.
- **Deep SVDD** (Ruff et al., 2018; distance-to-center in embedding space):
  first attempt collapsed (classic hypersphere collapse — embedding std
  dropped 300x over training); fixed with a `--min-embed-std` floor that
  hard-stops training at the first sign of collapse. Fixed version reached
  test ROC AUC 0.519, f1_ddos 0.011 — still weak, plausibly the same
  underlying sparsity-inversion cause, since the SVDD encoder is
  AE-pretrained and only gets a short pre-collapse training window to move
  away from that inherited representation.

Both are reported as genuine negative results for reconstruction- and
distance-based unsupervised detection on the FlowPic 0.3s representation on
this dataset — not tuning failures. Not included in the conference paper
(space) but documented here and in `WORK_LOG.md` for the full report and for
anyone extending this work.

## What's deliberately NOT included here

A third experiment — retraining the byte-dump encoding (not FlowPic) under
the same cluster-validated labels, to complete a full label-methodology ×
encoding ablation — was started the same night this report was written but
could not finish before a scheduled power outage. Rather than report an
undertrained checkpoint's numbers as if conclusive, that comparison is
omitted entirely; see `WORK_LOG.md`'s 2026-08-05/06 entries for status if
resumed later.
