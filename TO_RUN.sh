#!/usr/bin/env bash
# Status as of 2026-08-06 ~00:xx EEST. Nothing is running in the background
# right now -- the bytedump-validated-v1 training was deliberately killed
# (undertrained, dropped from the paper scope ahead of the 1:55am scheduled
# outage -- see HANDOVER_2026-08-06.md section 3 for how to resume it).
#
# GitHub push access is confirmed working (gh auth login done tonight).
# If a fresh session needs it again and `gh auth status` fails:
#   sudo snap install gh --classic
#   gh auth login
# (plain `apt install gh` doesn't work well on this box -- see
# HANDOVER_2026-08-06.md section 4)

# --- To resume the dropped 3-way encoding comparison (not urgent) ---
# See HANDOVER_2026-08-06.md section 3 for full detail. Quick version:
# cd /home/abed/Documents/huawei
# nohup bash scripts/train_resnext_per_second_0p3_validated_v1.sh \
#   > model/per_second_0p3_validated_v1/train_$(date +%Y-%m-%d_%H%M).log 2>&1 &
# disown

# --- Recompile the paper sources (recommended before submission) ---
# No pdflatex/texlive installed on this machine -- upload to Overleaf, or:
#   sudo apt install texlive-full   # large download, not done tonight
# Then from research/synshield_paper/:
#   pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
# From research/:
#   pdflatex main.tex && pdflatex main.tex   # (check existing build script/Makefile if any)

# --- Push to GitHub (works now, but sandbox still asks for confirmation
# on git push by default -- run yourself if asked) ---
# cd /home/abed/Documents/huawei && git push origin main
