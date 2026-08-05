#!/usr/bin/env bash
# Watch the official CICDDoS2019 CSV-01-12.zip download -- the only thing
# still actually running as of this writing (started 2026-08-04 23:24 EEST).
# No server-side resume support, so on a drop it fully restarts from byte 0
# automatically (up to 30 attempts) -- don't be alarmed if the file size
# drops back down between checks, that's the retry loop, not a bug.
# Ctrl-C to stop watching, doesn't stop the job:
watch -n 5 'ls -la /home/abed/Documents/huawei/dataset/CIC_official_labels/CSV-01-12.zip; echo; tail -5 /home/abed/Documents/huawei/dataset/CIC_official_labels/fetch.log'

# --- Below: not meant to run together with the above, copy individually ---
#
# Push commit f7405a9 to GitHub -- Claude Code's sandbox blocks `git push`
# without manual confirmation, so run this yourself:
# cd /home/abed/Documents/huawei && git push origin main

# --- CIC official-label CSV download detail ---
#
# Fetching CSVs/CSV-01-12.zip from cicresearch.ca to cross-check the 2,934
# flagged "false positive" flowpic windows in SAT-01-12-2018_0617/_0619
# against CICFlowMeter's authoritative per-flow Label column. Full detail:
# HANDOVER_2026-08-05.md section 2-3, or WORK_LOG.md 2026-08-04 ~22:15
# EEST onward.
#
# No Content-Length reported (chunked), no HTTP Range support (confirmed
# via curl -C -, error 33) -- dataset/CIC_official_labels/fetch_csv_01_12.sh
# restarts the whole download on any failure rather than resuming, up to 30
# attempts with a 30s backoff -- this is automatic, no action needed unless
# the process has actually died (check below). Session cookie
# (dataset/CIC_official_labels/cic_cookies.txt) expires ~23:20 EEST on
# 2026-08-05 -- if the log starts showing repeated curl failures after
# that, it needs a fresh export from Cookie-Editor (Netscape format) in
# your browser. Do NOT retry this on the private staging server
# (root@91.99.170.219) -- confirmed twice, that server's IP gets silently
# blocked (zero bytes, hangs) specifically on this endpoint.
#
# Just the fetch log:
# tail -f /home/abed/Documents/huawei/dataset/CIC_official_labels/fetch.log
#
# Check whether it's still alive -- if this comes back empty, relaunch with:
# nohup bash dataset/CIC_official_labels/fetch_csv_01_12.sh > /dev/null 2>&1 & disown
# ps aux | grep "[f]etch_csv_01_12\|[c]url.*CSV-01-12"

# --- Everything else from this session is DONE -- see HANDOVER_2026-08-05.md ---
# (supervised flowpic continuation run, autoencoder baseline, Deep SVDD
# baseline including the collapse-fix and the lr=5e-4 experiment) -- no
# commands needed, just read the handover for final numbers and what's
# still open.
